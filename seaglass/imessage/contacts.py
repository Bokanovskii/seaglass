"""Contacts resolution via the macOS Contacts framework (PyObjC).

PLAN.md §6 Phase 1: a `CNContact` already owns multiple phone numbers and
emails, so identity unification is system-maintained -- this is why
seaglass has no `person` table (DESIGN-NOTES.md §9, "A `person` table
mirroring contacts"). Contacts are loaded into memory once and resolved
live; nothing about a contact is ever persisted into `index.db`.

When the framework is denied, contacts are read read-only from the Contacts
database instead (see `ContactIndex.load`).
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re
import sqlite3
import unicodedata
from pathlib import Path
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
    from rapidfuzz import distance as rf_distance, fuzz, process as rf_process, utils as rf_utils
except ImportError:  # pragma: no cover
    fuzz = None
    rf_process = None
    rf_utils = None
    rf_distance = None

DEFAULT_REGION = "US"
ADDRESS_BOOK_DIR = Path.home() / "Library" / "Application Support" / "AddressBook"
_ADDRESS_BOOK_FILE = "AddressBook-v22.abcddb"


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



def _fold_name(raw: str) -> str:
    """Case- and accent-folded name, punctuation stripped, for exact
    comparison. "Kaya-Rose O'Neill" and "kaya rose oneill" fold alike."""
    decomposed = unicodedata.normalize("NFKD", raw or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^\w\s]+", " ", stripped)
    return " ".join(cleaned.lower().split())


def _display_name(given: object, family: object, nickname: object) -> str:
    return str(nickname or "") or f"{given or ''} {family or ''}".strip()


def _read_address_book(path: str, region: str) -> List[Contact]:
    con = sqlite3.connect(f"{Path(path).as_uri()}?mode=ro", uri=True)
    try:
        handles_by_owner: Dict[int, List[str]] = {}
        for query in (
            "SELECT ZOWNER, ZFULLNUMBER FROM ZABCDPHONENUMBER "
            "WHERE ZFULLNUMBER IS NOT NULL ORDER BY Z_PK",
            "SELECT ZOWNER, ZADDRESS FROM ZABCDEMAILADDRESS "
            "WHERE ZADDRESS IS NOT NULL ORDER BY Z_PK",
        ):
            for owner, raw in con.execute(query):
                handles_by_owner.setdefault(owner, []).append(_normalise_handle(str(raw), region))
        contacts: List[Contact] = []
        for pk, unique_id, given, family, nickname in con.execute(
            "SELECT Z_PK, ZUNIQUEID, ZFIRSTNAME, ZLASTNAME, ZNICKNAME FROM ZABCDRECORD ORDER BY Z_PK"
        ):
            display_name = _display_name(given, family, nickname)
            if not display_name:
                continue
            contacts.append(
                Contact(
                    identifier=str(unique_id or pk),
                    display_name=display_name,
                    handles=tuple(handles_by_owner.get(pk, ())),
                )
            )
        return contacts
    finally:
        con.close()


def _load_address_book_contacts(
    region: str = DEFAULT_REGION, root: Optional[Path] = None
) -> Optional[List[Contact]]:
    """Contacts read straight from every Contacts database, or None if none
    could be read (no Full Disk Access, not macOS, or an unknown schema)."""
    root = ADDRESS_BOOK_DIR if root is None else root
    paths = [os.path.join(root, _ADDRESS_BOOK_FILE)]
    paths.extend(sorted(glob.glob(os.path.join(glob.escape(str(root)), "Sources", "*", _ADDRESS_BOOK_FILE))))
    contacts: List[Contact] = []
    read_any = False
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            contacts.extend(_read_address_book(path, region))
        except sqlite3.Error:
            continue
        read_any = True
    return contacts if read_any else None


def contacts_authorization_status() -> int:
    """0=notDetermined 1=restricted 2=denied 3=authorized, -1 if unavailable."""
    if _Contacts is None:
        return -1
    try:
        return int(
            _Contacts.CNContactStore.authorizationStatusForEntityType_(
                _Contacts.CNEntityTypeContacts
            )
        )
    except Exception:  # noqa: BLE001 - status is advisory only
        return -1


def request_contacts_access(timeout: float = 60.0) -> dict:
    """Present the system Contacts prompt and report what happened.

    Returns `status` (a CNAuthorizationStatus) plus `granted` and
    `can_prompt`. `can_prompt` is False once the user has answered: macOS
    only ever shows the prompt once, so from then on the only route is the
    Settings pane, and the UI needs to know which of the two to offer.
    """
    status = contacts_authorization_status()
    if status == 3:
        return {'granted': True, 'status': status, 'can_prompt': False}
    if status != 0:
        # Already answered (denied/restricted) or unavailable: prompting
        # again is a silent no-op, so don't pretend otherwise.
        return {'granted': False, 'status': status, 'can_prompt': False}
    store = _Contacts.CNContactStore.alloc().init()
    _request_access(store, timeout=timeout)
    status = contacts_authorization_status()
    return {'granted': status == 3, 'status': status, 'can_prompt': False}


def _request_access(store, timeout: float = 30.0) -> None:
    """Ask for Contacts access and block until the user answers.

    Enumerating without asking just returns an empty list for an app whose
    grant is still undetermined -- which looks exactly like "you have no
    contacts", and is how every sender ended up rendered as a raw phone
    number the first time seaglass ran under its own app bundle identity
    (grants are per app identity, so an .app inherits none of the terminal's).

    Blocking matters: the completion handler fires on another thread, and
    warmup goes on to enumerate immediately, so returning early would race
    the user's answer and read an unauthorised store anyway.
    """
    import threading

    answered = threading.Event()

    def completion(granted, error):  # noqa: ANN001 - ObjC callback signature
        answered.set()

    try:
        store.requestAccessForEntityType_completionHandler_(
            _Contacts.CNEntityTypeContacts, completion
        )
        answered.wait(timeout)
    except Exception:  # noqa: BLE001 - a prompt failure must not break startup
        pass


def _load_framework_contacts(region: str) -> List[Contact]:
    if _Contacts is None:
        raise ContactsUnavailableError(
            "pyobjc-framework-Contacts is not installed or unavailable on this platform"
        )
    status = contacts_authorization_status()
    # 0=notDetermined 1=restricted 2=denied 3=authorized (values per CNAuthorizationStatus)
    if status not in (0, 3):
        raise ContactsUnavailableError(
            f"Contacts access not authorized (status={status}); grant access in "
            "System Settings > Privacy & Security > Contacts, or grant Full Disk Access"
        )
    store = _Contacts.CNContactStore.alloc().init()
    if status == 0:
        # Best effort only, with a short timeout: during warmup there is
        # usually no Cocoa event loop yet to present the prompt on (see
        # request_contacts_access), so this must not stall startup.
        _request_access(store, timeout=2.0)
        status = contacts_authorization_status()
        if status != 3:
            raise ContactsUnavailableError(f"Contacts access not granted (status={status})")
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
        display_name = _display_name(contact.givenName(), contact.familyName(), contact.nickname())
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
    return contacts


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
        """Load every contact, from the framework or else the database.

        The database fallback is for processes macOS never asks about
        Contacts: an unbundled interpreter run by launchd (a headless
        gateway, say) is denied by service policy, with no prompt and no
        Settings toggle to grant. Full Disk Access, which chat.db already
        needs, does cover the database files.
        """
        try:
            contacts = _load_framework_contacts(region)
        except ContactsUnavailableError:
            contacts = _load_address_book_contacts(region)
            if contacts is None:
                raise
        return cls(contacts, region=region)

    def resolve_handle(self, handle: str) -> Optional[str]:
        """handle (as stored in chat.db `handle.id`) -> display name, or None."""
        return self._by_handle.get(_normalise_handle(handle, self._region), None) and \
            self._by_handle[_normalise_handle(handle, self._region)].display_name

    def fuzzy_match(self, query_tokens: str, threshold: float = 88.0) -> List[Contact]:
        """Fuzzy name -> contact match, used only above `threshold`
        (PLAN.md §6 Phase 4: "prefer omitting [the filter] to guessing").

        Matching is case-insensitive. rapidfuzz applies no processor by
        default, so without one "kaya" scored 0 against "Kaya Doe" while
        "Kaya" scored 86 -- people type their friends' names in lower case
        constantly, and every such query silently lost its person filter.
        """
        if rf_process is None or not self._names:
            return []
        matches = rf_process.extract(
            query_tokens,
            self._names,
            scorer=fuzz.WRatio,
            processor=rf_utils.default_process,
            score_cutoff=threshold,
            limit=5,
        )
        matched_names = {name for name, score, _ in matches}
        return [c for c in self._contacts if c.display_name in matched_names]

    def handle_ids_for_similar_given_name(self, name: str) -> List[str]:
        """Handles for contacts whose *given* name is one typo away.

        Fuzzy WRatio against the full display name cannot separate these:
        "Kaay" scores 77 against "Kaya Makivic" while "my mom" scores 85
        against "Rogers Mom", so no threshold admits the typo without
        admitting the wrong answer. Comparing against the given name with a
        bounded edit distance does separate them -- one transposition, same
        first letter, long enough that the edit is not most of the word.
        """
        wanted = _fold_name(name)
        if len(wanted) < 4 or " " in wanted or rf_distance is None:
            return []
        handles: List[str] = []
        for contact in self._contacts:
            folded = _fold_name(contact.display_name)
            if not folded:
                continue
            given = folded.split()[0]
            if len(given) < 4 or given[0] != wanted[0]:
                continue
            # Optimal string alignment: a transposition is one edit, which
            # is exactly the mistake people make typing a name.
            if rf_distance.OSA.distance(wanted, given, score_cutoff=1) <= 1:
                handles.extend(contact.handles)
        return handles

    def handle_ids_for_names(self, query_tokens: str, threshold: float = 88.0) -> List[str]:
        """Convenience: fuzzy name match -> the handle strings to filter on."""
        handles: List[str] = []
        for contact in self.fuzzy_match(query_tokens, threshold):
            handles.extend(contact.handles)
        return handles

    def handle_ids_for_exact_name(self, name: str) -> List[str]:
        """Handles for contacts whose name matches `name` exactly, ignoring
        case and punctuation.

        Used for name candidates that carry no capitalization evidence
        ("from kaya"), where fuzzy matching is too loose to be safe -- a
        lower-cased "the trip" partial-matches real contact names well above
        the fuzzy threshold. An exact hit has no such failure mode.

        Only the whole name and the *given* name count. Matching any token
        read "from my mom" as the contact "Rogers Mom" -- somebody else's
        mother, confidently and wrongly.
        """
        wanted = _fold_name(name)
        if not wanted:
            return []
        handles: List[str] = []
        for contact in self._contacts:
            folded = _fold_name(contact.display_name)
            if not folded:
                continue
            given = folded.split()[0]
            if wanted == folded or (len(given) > 2 and wanted == given):
                handles.extend(contact.handles)
        return handles
