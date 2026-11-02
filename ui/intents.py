"""Detects confirmations, cancellations and booking requests in a message."""

_CONFIRM_WORDS = {
    # English
    "confirm",
    "yes",
    "ok",
    "okay",
    "go",
    "proceed",
    "sure",
    "do",
    "approve",
    "yep",
    "yup",
    "right",
    "correct",
    "affirmative",
    # Italian
    "sì",
    "si",
    "conferma",
    "confermo",
    "vai",
    "procedi",
    "approva",
    "certo",
    "esatto",
    "perfetto",
    # German
    "ja",
    "bestätigen",
    "weiter",
    "sicher",
    # Spanish
    "sí",
    "confirmar",
    "proceder",
    "claro",
    # Slovenian
    "da",
    "potrdi",
}
_CANCEL_WORDS = {
    # English
    "cancel",
    "no",
    "stop",
    "abort",
    "reject",
    "nope",
    "nah",
    "refuse",
    # Italian
    "annulla",
    "ferma",
    "rifiuta",
    # German
    "nein",
    "abbrechen",
    "stopp",
    # Spanish
    "cancelar",
    "rechazar",
    # Slovenian
    "ne",
    "prekliči",
}
_CONFIRM_PHRASES = [
    "go ahead",
    "do it",
    "go for it",
    "sure thing",
    "sounds good",
    "let's do it",
    "va bene",
    "va benissimo",
    "d'accordo",
    "andiamo",
]
_CANCEL_PHRASES = ["don't do it", "do not", "don't proceed", "abort mission"]


def _detect_hitl_intent(text: str):
    """
    Returns 'confirmed', 'rejected', or None (ambiguous).
    Checks single-word matches first, then short-phrase matches.
    """
    stripped = text.strip().rstrip(".,!?")
    lower = stripped.lower()
    words = set(lower.split())

    # Short messages: treat the whole text as a single keyword
    if len(words) <= 3:
        if words & _CONFIRM_WORDS:
            return "confirmed"
        if words & _CANCEL_WORDS:
            return "rejected"

    for phrase in _CONFIRM_PHRASES:
        if phrase in lower:
            return "confirmed"
    for phrase in _CANCEL_PHRASES:
        if phrase in lower:
            return "rejected"

    return None


_BOOKING_INTENT_PHRASES = [
    # Italian — deliberately "vorrei/voglio + verb" combinations rather than
    # bare verbs like "installare"/"comprare": the bare form also appears in
    # purely informational questions ("come si installa...", "quanto costa
    # comprare...") that must NOT pop the booking form. "sopralluogo" and
    # "preventivo" are bare (no verb-gating) because, unlike a verb, the noun
    # itself is booking-specific enough on its own not to false-positive on
    # informational questions.
    "prenotare",
    "prenotazione",
    "sopralluogo",
    "preventivo",
    "richiedere un sopralluogo",
    "vorrei installare",
    "voglio installare",
    "vorrei acquistare",
    "voglio acquistare",
    "vorrei comprare",
    "voglio comprare",
    "vorrei ordinare",
    "voglio ordinare",
    "vorrei procedere",
    "voglio procedere",
    # English — "survey"/"quote"/"estimate" are bare nouns (same reasoning as
    # the Italian "sopralluogo"/"preventivo" above): a phrase-only net like
    # "site survey"/"request a survey" misses everyday phrasings such as
    # "I'd like to ask for a survey" or "can I get a survey done".
    "book an installation",
    "site visit",
    "site survey",
    "get a quote",
    "request a survey",
    "survey",
    "quote",
    "estimate",
    "i want to install",
    "i'd like to install",
    "i want to buy",
    "i'd like to buy",
    "i want to order",
    "i'd like to order",
    "i want to proceed",
    "i'd like to proceed",
    # German
    "möchte installieren",
    "will installieren",
    "möchte kaufen",
    "will kaufen",
    "möchte bestellen",
    "will bestellen",
    "besichtigung",
    "kostenvoranschlag",
    "angebot",
    # Spanish
    "quiero instalar",
    "me gustaría instalar",
    "quiero comprar",
    "me gustaría comprar",
    "quiero pedir",
    "me gustaría pedir",
    "visita técnica",
    "presupuesto",
    # Slovenian
    "želim namestiti",
    "želim kupiti",
    "želim naročiti",
    "ogled",
    "ponudba",
]


def _detect_booking_intent(text: str) -> bool:
    """
    Lightweight keyword check for booking/survey intent in a client's organic
    message — used to surface the structured booking form instead of letting
    the agent collect the same fields one question at a time in plain chat.
    """
    lower = text.strip().lower()
    return any(phrase in lower for phrase in _BOOKING_INTENT_PHRASES)
