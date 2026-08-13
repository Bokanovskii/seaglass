

class TestSenderFiltering:
    def _session(self):
        from seaglass.search.hydrate import HydratedMessage, HydratedSession

        def message(mid, handle):
            return HydratedMessage(
                message_id=mid, ts=float(mid), is_from_me=handle is None,
                sender=handle, text=f'm{mid}', has_attachment=False, handle=handle,
            )

        return HydratedSession(
            chat_id=1, day='2026-08-12', score=1.0,
            hit_messages=[message(1, '+1555'), message(2, '+1666'), message(3, None)],
            context_messages=[message(4, '+1666')],
        )

    def test_only_the_named_senders_messages_stay_hits(self):
        from seaglass.app.engine import _filter_by_sender

        kept = _filter_by_sender([self._session()], ['+1555'])
        assert [m.message_id for m in kept[0].hit_messages] == [1]

    def test_the_others_are_demoted_to_context_not_dropped(self):
        from seaglass.app.engine import _filter_by_sender

        kept = _filter_by_sender([self._session()], ['+1555'])
        assert {m.message_id for m in kept[0].context_messages} == {2, 3, 4}

    def test_a_session_the_sender_never_spoke_in_is_dropped(self):
        from seaglass.app.engine import _filter_by_sender

        assert _filter_by_sender([self._session()], ['+1999']) == []

    def test_handle_matching_ignores_case(self):
        from seaglass.app.engine import _filter_by_sender

        from dataclasses import replace

        session = self._session()
        session.hit_messages[0] = replace(session.hit_messages[0], handle='Kaya@Example.com')
        kept = _filter_by_sender([session], ['kaya@example.com'])
        assert [m.message_id for m in kept[0].hit_messages] == [1]


class TestBrowseOrdering:
    """Browsing means newest-first, and must not be decided by chunk count."""

    def _rows(self):
        # chat 1 has two chunks on an older day; chat 2 has one newer chunk.
        day = 86400.0
        return [(11, 1, 1_000_000.0), (12, 1, 1_000_000.0 + 60), (21, 2, 1_000_000.0 + 5 * day)]

    def test_a_newer_single_chunk_day_outranks_an_older_busy_one(self):
        from seaglass.search.rank import RankedChunk, order_sessions

        rows = self._rows()
        chunks = [
            RankedChunk(chunk_id=cid, chat_id=chat, start_ts=ts, rerank_score=float(len(rows) - i))
            for i, (cid, chat, ts) in enumerate(rows)
        ]
        ordered = order_sessions(chunks, head_size=5, recent_slots=0)
        newest = {cid: ts for cid, _chat, ts in rows}
        ordered.sort(
            key=lambda s: max((newest.get(c, 0.0) for c in s.hit_chunk_ids), default=0.0),
            reverse=True,
        )
        assert ordered[0].chat_id == 2

    def test_the_pseudo_scores_alone_would_have_got_this_wrong(self):
        # Documents *why* the explicit sort exists: the sigmoid saturates,
        # so summing it ranks by how many chunks a day has.
        from seaglass.search.rank import RankedChunk, order_sessions

        rows = self._rows()
        chunks = [
            RankedChunk(chunk_id=cid, chat_id=chat, start_ts=ts, rerank_score=float(len(rows) - i))
            for i, (cid, chat, ts) in enumerate(rows)
        ]
        assert order_sessions(chunks, head_size=5, recent_slots=0)[0].chat_id == 1


class TestSenderFilterExcludesMyOwnMessages:
    def test_outgoing_messages_stamped_with_the_recipients_handle_are_not_hits(self):
        # chat.db puts the *other* party's handle_id on outgoing 1:1
        # messages, so handle alone matches both halves of the exchange.
        from seaglass.app.engine import _filter_by_sender
        from seaglass.search.hydrate import HydratedMessage, HydratedSession

        def message(mid, from_me):
            return HydratedMessage(
                message_id=mid, ts=float(mid), is_from_me=from_me, sender=None,
                text='x', has_attachment=False, handle='+1555',
            )

        session = HydratedSession(
            chat_id=1, day='d', score=1.0,
            hit_messages=[message(1, False), message(2, True)], context_messages=[],
        )
        kept = _filter_by_sender([session], ['+1555'])
        assert [m.message_id for m in kept[0].hit_messages] == [1]


class TestBrowseKeepsTheNewestHits:
    def _sessions(self, *sizes, base=0):
        from seaglass.search.hydrate import HydratedMessage, HydratedSession

        sessions = []
        mid = base
        for chat_id, size in enumerate(sizes, start=1):
            messages = []
            for _ in range(size):
                mid += 1
                messages.append(
                    HydratedMessage(message_id=mid, ts=float(mid), is_from_me=False,
                                    sender='A', text=f'm{mid}', has_attachment=False,
                                    handle='+1555')
                )
            sessions.append(
                HydratedSession(chat_id=chat_id, day=f'd{chat_id}', score=1.0,
                                hit_messages=messages, context_messages=[])
            )
        return sessions

    def test_hits_come_back_newest_first(self):
        # The payload declares `ordering: recent`; a caller reading the top
        # must get the latest message, not the oldest of the newest day.
        from seaglass.app.engine import _newest_hits_only

        kept = _newest_hits_only(self._sessions(30), per_session=10)[0]
        assert [m.message_id for m in kept.hit_messages] == list(range(30, 20, -1))

    def test_the_budget_is_global_not_per_session(self):
        # Per-session allocation cut 15 of the 20 most recent messages in
        # favour of older ones from a week ago (recall@20 was 0.64).
        from seaglass.app.engine import _newest_hits_only

        # Session 2 holds the newest 25 messages; session 1 the oldest 25.
        sessions = self._sessions(25, 25)
        kept = _newest_hits_only(sessions, budget=20)
        assert sum(len(s.hit_messages) for s in kept) == 20
        newest = [m.message_id for s in kept for m in s.hit_messages]
        assert set(newest) == set(range(31, 51))

    def test_a_session_that_wins_no_budget_is_dropped(self):
        from seaglass.app.engine import _newest_hits_only

        kept = _newest_hits_only(self._sessions(10, 10), budget=5)
        assert [s.chat_id for s in kept] == [2]

    def test_the_trimmed_messages_become_context_not_losses(self):
        from seaglass.app.engine import _newest_hits_only

        kept = _newest_hits_only(self._sessions(30), per_session=10)[0]
        assert len(kept.hit_messages) + len(kept.context_messages) == 30

    def test_a_short_session_keeps_every_hit(self):
        from seaglass.app.engine import _newest_hits_only

        kept = _newest_hits_only(self._sessions(4), per_session=10)[0]
        assert len(kept.hit_messages) == 4


class TestFirstPersonFilter:
    def _session(self):
        from seaglass.search.hydrate import HydratedMessage, HydratedSession

        messages = [
            HydratedMessage(message_id=1, ts=1.0, is_from_me=True, sender=None,
                            text='mine', has_attachment=False, handle='+1555'),
            HydratedMessage(message_id=2, ts=2.0, is_from_me=False, sender='K',
                            text='theirs', has_attachment=False, handle='+1555'),
        ]
        return [HydratedSession(chat_id=1, day='d', score=1.0,
                                hit_messages=messages, context_messages=[])]

    def test_it_keeps_only_my_messages(self):
        from seaglass.app.engine import _filter_by_from_me

        kept = _filter_by_from_me(self._session(), True)[0]
        assert [m.message_id for m in kept.hit_messages] == [1]

    def test_the_other_side_becomes_context_not_a_loss(self):
        from seaglass.app.engine import _filter_by_from_me

        kept = _filter_by_from_me(self._session(), True)[0]
        assert [m.message_id for m in kept.context_messages] == [2]


class TestBrowseOrdersSessionsByRecency:
    def test_sessions_come_back_newest_first(self):
        from seaglass.app.engine import _newest_hits_only
        from seaglass.search.hydrate import HydratedMessage, HydratedSession

        def session(chat_id, stamps):
            messages = [
                HydratedMessage(message_id=int(t), ts=float(t), is_from_me=False,
                                sender='A', text='x', has_attachment=False, handle='+1')
                for t in stamps
            ]
            return HydratedSession(chat_id=chat_id, day='d', score=1.0,
                                   hit_messages=messages, context_messages=[])

        kept = _newest_hits_only([session(1, [10, 11]), session(2, [20, 21])])
        kept.sort(key=lambda s: max(m.ts for m in s.hit_messages), reverse=True)
        assert [s.chat_id for s in kept] == [2, 1]
        assert [m.ts for s in kept for m in s.hit_messages] == [21, 20, 11, 10]
