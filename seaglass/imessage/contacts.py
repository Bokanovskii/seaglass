"""Contacts resolution via the macOS Contacts framework (PyObjC).

PLAN.md §6 Phase 1: a `CNContact` already owns multiple phone numbers and
emails, so identity unification is system-maintained -- this is why
seaglass has no `person` table (DESIGN-NOTES.md §9, "A `person` table
mirroring contacts"). Contacts are loaded into memory once and resolved
live; nothing about a contact is ever persisted into `index.db`.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from typing import Dict, List, Optional

try:
    import Contacts as _Contacts  # PyObjC framework binding
except ImportError:  # pragma: no cover - exercised only off macOS or without pyobjc
    _Contacts = None

try:
    import phonenumbers
except ImportError:  # pragma: no cover
    phonenumbers = None

try:
    from rapidfuzz import fuzz, process as rf_process
except ImportError:  # pragma: no cover
    fuzz = None
    rf_process = None

DEFAULT_REGION = "US"


class ContactsUnavailableError(RuntimeError):
    """Raised when the Contacts framework binding or authorization is missing."""


@dataclasses.dataclass(frozen=True)
class Contact:
    identifier: str  # CNContact.identifier
    display_name: str
    handles: tuple  # normalised phone/email strings, this contact's identifiers


def _normalise_phone(raw: str, region: str = DEFAULT_REGION) -> Optional[str]:
    """E.164 normalisation. PLAN.md §6 Phase 1: "fiddlier than it looks."""
    if phonenumbers is None:
        return raw
    try:
        parsed = phonenumbers.parse(raw, region)
        if not phonenumbers.is_valid_number(parsed):
            return raw
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        return raw


def _normalise_handle(raw: str, region: str = DEFAULT_REGION) -> str:
    raw = raw.strip()
    if "@" in raw:
        return raw.lower()
    normalised = _normalise_phone(raw, region)
    return normalised or raw


class ContactIndex:
    """In-memory contact roster, loaded once from the Contacts framework.

    Two lookups are supported, matching the two directions PLAN.md §6
    Phase 1 describes:

    * `identifier (E.164 / email) -> display_name` -- for hydration/display.
    * fuzzy name -> handle_id set -- for the query-parser's participant
      filter (search/parse.py), applied only above a confidence threshold.
    """

    def __init__(self, contacts: List[Contact], region: str = DEFAULT_REGION):
        self._contacts = contacts
        self._region = region
        self._by_handle: Dict[str, Contact] = {}
        for contact in contacts:
            for handle in contact.handles:
                self._by_handle[handle] = contact
        self._names = [c.display_name for c in contacts if c.display_name]

    @classmethod
    def load(cls, region: str = DEFAULT_REGION) -> "ContactIndex":
        if _Contacts is None:
            raise ContactsUnavailableError(
                "pyobjc-framework-Contacts is not installed or unavailable on this platform"
            )
        status = _Contacts.CNContactStore.authorizationStatusForEntityType_(
            _Contacts.CNEntityTypeContacts
        )
        # 0=notDetermined 1=restricted 2=denied 3=authorized (values per CNAuthorizationStatus)
        if status not in (0, 3):
            raise ContactsUnavailableError(
                f"Contacts access not authorized (status={status}); grant access in "
                "System Settings > Privacy & Security > Contacts"
            )
        store = _Contacts.CNContactStore.alloc().init()
        keys = [
            _Contacts.CNContactGivenNameKey,
            _Contacts.CNContactFamilyNameKey,
            _Contacts.CNContactNicknameKey,
            _Contacts.CNContactPhoneNumbersKey,
            _Contacts.CNContactEmailAddressesKey,
            _Contacts.CNContactIdentifierKey,
        ]
        request = _Contacts.CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)
        contacts: List[Contact] = []
        collected: List[object] = []

        def handler(contact, stop):
            collected.append(contact)

        ok, error = store.enumerateContactsWithFetchRequest_error_usingBlock_(
            request, None, handler
        )
        if not ok:
            raise ContactsUnavailableError(f"enumerateContactsWithFetchRequest failed: {error}")

        for contact in collected:
            given = str(contact.givenName() or "")
            family = str(contact.familyName() or "")
            nickname = str(contact.nickname() or "")
            display_name = nickname or f"{given} {family}".strip()
            if not display_name:
                continue
            handles = []
            for labeled in contact.phoneNumbers():
                value = labeled.value()
                digits = str(value.stringValue())
                handles.append(_normalise_handle(digits, region))
            for labeled in contact.emailAddresses():
                handles.append(_normalise_handle(str(labeled.value()), region))
            contacts.append(
                Contact(
                    identifier=str(contact.identifier()),
                    display_name=display_name,
                    handles=tuple(handles),
                )
            )
        return cls(contacts, region=region)

    def resolve_handle(self, handle: str) -> Optional[str]:
        """handle (as stored in chat.db `handle.id`) -> display name, or None."""
        return self._by_handle.get(_normalise_handle(handle, self._region), None) and \
            self._by_handle[_normalise_handle(handle, self._region)].display_name

    def fuzzy_match(self, query_tokens: str, threshold: float = 88.0) -> List[Contact]:
        """Fuzzy name -> contact match, used only above `threshold`
        (PLAN.md §6 Phase 4: "prefer omitting [the filter] to guessing").
        """
        if rf_process is None or not self._names:
            return []
        matches = rf_process.extract(
            query_tokens, self._names, scorer=fuzz.WRatio, score_cutoff=threshold, limit=5
        )
        matched_names = {name for name, score, _ in matches}
        return [c for c in self._contacts if c.display_name in matched_names]

    def handle_ids_for_names(self, query_tokens: str, threshold: float = 88.0) -> List[str]:
        """Convenience: fuzzy name match -> the handle strings to filter on."""
        handles: List[str] = []
        for contact in self.fuzzy_match(query_tokens, threshold):
            handles.extend(contact.handles)
        return handles
