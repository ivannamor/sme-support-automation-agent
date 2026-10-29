"""
Shared Streamlit UI helpers used across client, installer, and manager pages.

All per-chat session state is namespaced by role so that switching between
pages does not bleed conversation history or thread IDs across roles.
Global state (role, username, ux_lang) stays un-prefixed.
"""

import contextlib
import datetime
import logging
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote as _url_quote

import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

# TODO PR 11: drop with the app/ restructure.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.assistant import TurnResult, initialize_agent
from backend.company_config import COMPANY_CONTEXT
from backend.database import (
    get_config,
    get_costs,
    get_inventory,
    get_product,
    get_ticket_log,
    log_usage,
    set_config,
    ticket_code,
)
from backend.llm_provider import DEFAULT_MODEL, OPENROUTER_MODELS, calculate_cost
from backend.log_context import username_var, verified_client_email_var
from backend.qa_cache import lookup as _cache_lookup
from backend.qa_cache import submit_for_review as qa_submit

# ── translations ──────────────────────────────────────────────────────────────

_LANGS = ["English", "Italiano", "Deutsch", "Español", "Slovenščina"]

_TR: dict[str, dict[str, str]] = {
    # Sidebar / navigation
    "logout": {
        "English": "Logout",
        "Italiano": "Esci",
        "Deutsch": "Abmelden",
        "Español": "Cerrar sesión",
        "Slovenščina": "Odjava",
    },
    "clear": {
        "English": "Clear Conversation",
        "Italiano": "Cancella conversazione",
        "Deutsch": "Gespräch löschen",
        "Español": "Borrar conversación",
        "Slovenščina": "Počisti pogovor",
    },
    "view_catalogue": {
        "English": "View Catalogue PDF",
        "Italiano": "Visualizza catalogo PDF",
        "Deutsch": "Katalog PDF anzeigen",
        "Español": "Ver catálogo PDF",
        "Slovenščina": "Prikaži katalog PDF",
    },
    "language_label": {
        "English": "Language",
        "Italiano": "Lingua",
        "Deutsch": "Sprache",
        "Español": "Idioma",
        "Slovenščina": "Jezik",
    },
    "voice_label": {
        "English": "Voice Mode",
        "Italiano": "Modalità vocale",
        "Deutsch": "Sprachmodus",
        "Español": "Modo de voz",
        "Slovenščina": "Glasovni način",
    },
    "model_label": {
        "English": "Model",
        "Italiano": "Modello",
        "Deutsch": "Modell",
        "Español": "Modelo",
        "Slovenščina": "Model",
    },
    # HITL
    "confirm": {
        "English": "Confirm",
        "Italiano": "Conferma",
        "Deutsch": "Bestätigen",
        "Español": "Confirmar",
        "Slovenščina": "Potrdi",
    },
    "cancel": {
        "English": "Cancel",
        "Italiano": "Annulla",
        "Deutsch": "Abbrechen",
        "Español": "Cancelar",
        "Slovenščina": "Prekliči",
    },
    "action_required": {
        "English": "Action required",
        "Italiano": "Azione richiesta",
        "Deutsch": "Aktion erforderlich",
        "Español": "Acción requerida",
        "Slovenščina": "Zahtevano dejanje",
    },
    # Answer banners
    "cache_hit": {
        "English": "Answered from the knowledge cache.",
        "Italiano": "Risposta dalla cache di conoscenza.",
        "Deutsch": "Aus dem Wissenscache beantwortet.",
        "Español": "Respondido desde la caché de conocimiento.",
        "Slovenščina": "Odgovor iz baze znanja.",
    },
    "general_llm": {
        "English": "This answer is based on general AI knowledge, not the product documentation. Verify with a specialist for safety-critical decisions.",
        "Italiano": "Questa risposta si basa sulla conoscenza generale dell'AI, non sulla documentazione di prodotto. Verificare con uno specialista per decisioni critiche.",
        "Deutsch": "Diese Antwort basiert auf allgemeinem KI-Wissen, nicht auf der Produktdokumentation. Bei sicherheitskritischen Entscheidungen Experten hinzuziehen.",
        "Español": "Esta respuesta se basa en el conocimiento general de la IA, no en la documentación del producto. Consulte a un especialista para decisiones críticas.",
        "Slovenščina": "Ta odgovor temelji na splošnem znanju AI, ne na dokumentaciji izdelka. Za varnostno kritične odločitve se posvetujte s strokovnjakom.",
    },
    "open_catalogue": {
        "English": "Open in catalogue — page",
        "Italiano": "Apri nel catalogo — pag.",
        "Deutsch": "Im Katalog öffnen — S.",
        "Español": "Abrir en el catálogo — pág.",
        "Slovenščina": "Odpri v katalogu — str.",
    },
    # Voice
    "recording_hint": {
        "English": "Click to record",
        "Italiano": "Fai clic per registrare",
        "Deutsch": "Zum Aufnehmen klicken",
        "Español": "Haz clic para grabar",
        "Slovenščina": "Kliknite za snemanje",
    },
    "transcribing": {
        "English": "Transcribing...",
        "Italiano": "Trascrizione in corso...",
        "Deutsch": "Transkribieren...",
        "Español": "Transcribiendo...",
        "Slovenščina": "Prepisovanje...",
    },
    "replay": {
        "English": "Replay audio",
        "Italiano": "Riproduci audio",
        "Deutsch": "Audio wiederholen",
        "Español": "Repetir audio",
        "Slovenščina": "Ponovi zvok",
    },
    "audio_error": {
        "English": "Audio unavailable.",
        "Italiano": "Audio non disponibile.",
        "Deutsch": "Audio nicht verfügbar.",
        "Español": "Audio no disponible.",
        "Slovenščina": "Zvok ni na voljo.",
    },
    # Page titles / subtitles
    "client_title": {
        "English": "Client Portal",
        "Italiano": "Portale Cliente",
        "Deutsch": "Kundenportal",
        "Español": "Portal del Cliente",
        "Slovenščina": "Portal za stranke",
    },
    "client_sub": {
        "English": "Ask about products, availability, and request installation bookings.",
        "Italiano": "Chiedi informazioni su prodotti, disponibilità e prenota installazioni.",
        "Deutsch": "Fragen zu Produkten, Verfügbarkeit und Installationsbuchungen.",
        "Español": "Consulta productos, disponibilidad y solicita instalaciones.",
        "Slovenščina": "Povprašajte o izdelkih, razpoložljivosti in naročite montažo.",
    },
    "installer_title": {
        "English": "Installer Portal",
        "Italiano": "Portale Installatore",
        "Deutsch": "Installateurportal",
        "Español": "Portal del Instalador",
        "Slovenščina": "Portal za monterje",
    },
    "installer_sub": {
        "English": "Access technical documentation and report site issues.",
        "Italiano": "Accedi alla documentazione tecnica e segnala problemi sul cantiere.",
        "Deutsch": "Technische Dokumentation abrufen und Probleme vor Ort melden.",
        "Español": "Accede a documentación técnica y reporta incidencias en obra.",
        "Slovenščina": "Dostopajte do tehnične dokumentacije in poročajte o težavah.",
    },
    "manager_title": {
        "English": "Manager Portal",
        "Italiano": "Portale Manager",
        "Deutsch": "Managerportal",
        "Español": "Portal del Gerente",
        "Slovenščina": "Portal za vodjo",
    },
    "manager_sub": {
        "English": "Manage inventory, book appointments, answer technical questions...",
        "Italiano": "Gestisci l'inventario, prenota appuntamenti, rispondi a domande tecniche...",
        "Deutsch": "Bestand verwalten, Termine buchen, technische Fragen beantworten...",
        "Español": "Gestiona el inventario, reserva citas, responde preguntas técnicas...",
        "Slovenščina": "Upravljajte zalogo, rezervirajte termine, odgovarjajte na tehnična vprašanja...",
    },
    "assigned_jobs": {
        "English": "Assigned Jobs",
        "Italiano": "Lavori assegnati",
        "Deutsch": "Zugewiesene Aufträge",
        "Español": "Trabajos asignados",
        "Slovenščina": "Dodeljeni posli",
    },
    "no_jobs": {
        "English": "No jobs assigned yet.",
        "Italiano": "Nessun lavoro assegnato.",
        "Deutsch": "Keine Aufträge zugeteilt.",
        "Español": "Sin trabajos asignados.",
        "Slovenščina": "Še ni dodeljenih poslov.",
    },
    "assistant_label": {
        "English": "AI Assistant",
        "Italiano": "Assistente AI",
        "Deutsch": "KI-Assistent",
        "Español": "Asistente de IA",
        "Slovenščina": "AI pomočnik",
    },
    "go_login": {
        "English": "Go to Login",
        "Italiano": "Vai al login",
        "Deutsch": "Zum Login",
        "Español": "Ir al inicio de sesión",
        "Slovenščina": "Pojdi na prijavo",
    },
    # Booking form
    "booking_form_title": {
        "English": "Site survey request",
        "Italiano": "Richiesta di sopralluogo",
        "Deutsch": "Besichtigungsanfrage",
        "Español": "Solicitud de visita técnica",
        "Slovenščina": "Zahteva za ogled",
    },
    "booking_form_name": {
        "English": "Full name",
        "Italiano": "Nome completo",
        "Deutsch": "Vollständiger Name",
        "Español": "Nombre completo",
        "Slovenščina": "Polno ime",
    },
    "booking_form_address": {
        "English": "Installation address",
        "Italiano": "Indirizzo di installazione",
        "Deutsch": "Installationsadresse",
        "Español": "Dirección de instalación",
        "Slovenščina": "Naslov montaže",
    },
    "booking_form_model": {
        "English": "Model",
        "Italiano": "Modello",
        "Deutsch": "Modell",
        "Español": "Modelo",
        "Slovenščina": "Model",
    },
    "booking_form_quantity": {
        "English": "Quantity",
        "Italiano": "Quantità",
        "Deutsch": "Menge",
        "Español": "Cantidad",
        "Slovenščina": "Količina",
    },
    "booking_form_date": {
        "English": "Date",
        "Italiano": "Data",
        "Deutsch": "Datum",
        "Español": "Fecha",
        "Slovenščina": "Datum",
    },
    "booking_form_time": {
        "English": "Time",
        "Italiano": "Ora",
        "Deutsch": "Uhrzeit",
        "Español": "Hora",
        "Slovenščina": "Ura",
    },
    "booking_form_month": {
        "English": "Month",
        "Italiano": "Mese",
        "Deutsch": "Monat",
        "Español": "Mes",
        "Slovenščina": "Mesec",
    },
    "booking_form_day": {
        "English": "Day",
        "Italiano": "Giorno",
        "Deutsch": "Tag",
        "Español": "Día",
        "Slovenščina": "Dan",
    },
    "booking_form_hour": {
        "English": "Hour",
        "Italiano": "Ora",
        "Deutsch": "Stunde",
        "Español": "Hora",
        "Slovenščina": "Ura",
    },
    "booking_form_minute": {
        "English": "Minute",
        "Italiano": "Minuti",
        "Deutsch": "Minute",
        "Español": "Minuto",
        "Slovenščina": "Minuta",
    },
    "booking_form_asap_button": {
        "English": "As soon as possible",
        "Italiano": "Il prima possibile",
        "Deutsch": "So schnell wie möglich",
        "Español": "Lo antes posible",
        "Slovenščina": "Čim prej",
    },
    "booking_form_email": {
        "English": "Email",
        "Italiano": "Email",
        "Deutsch": "E-Mail",
        "Español": "Correo electrónico",
        "Slovenščina": "E-pošta",
    },
    "booking_form_phone": {
        "English": "Phone",
        "Italiano": "Telefono",
        "Deutsch": "Telefon",
        "Español": "Teléfono",
        "Slovenščina": "Telefon",
    },
    "booking_form_submit": {
        "English": "Send request",
        "Italiano": "Invia richiesta",
        "Deutsch": "Anfrage senden",
        "Español": "Enviar solicitud",
        "Slovenščina": "Pošlji zahtevo",
    },
    "booking_form_missing": {
        "English": "Missing required fields",
        "Italiano": "Campi obbligatori mancanti",
        "Deutsch": "Pflichtfelder fehlen",
        "Español": "Faltan campos obligatorios",
        "Slovenščina": "Manjkajo obvezna polja",
    },
    "booking_form_invalid_date": {
        "English": "Invalid date — please check the day/month combination",
        "Italiano": "Data non valida — controlla la combinazione giorno/mese",
        "Deutsch": "Ungültiges Datum — bitte Tag/Monat prüfen",
        "Español": "Fecha no válida — comprueba la combinación de día/mes",
        "Slovenščina": "Neveljaven datum — preverite kombinacijo dneva/meseca",
    },
    "booking_intent_redirect": {
        "English": "I've opened the site survey request form below — fill it in (or describe your request in your own words and let AI fill it in for you).",
        "Italiano": "Ho aperto qui sotto il modulo di richiesta sopralluogo — compilalo (oppure descrivi la tua richiesta con parole tue e lascia che l'AI lo compili per te).",
        "Deutsch": "Ich habe unten das Formular für die Besichtigungsanfrage geöffnet — füllen Sie es aus (oder beschreiben Sie Ihre Anfrage mit eigenen Worten und lassen Sie es von der KI ausfüllen).",
        "Español": "He abierto el formulario de solicitud de visita técnica abajo — complétalo (o describe tu solicitud con tus propias palabras y deja que la IA lo complete por ti).",
        "Slovenščina": "Spodaj sem odprl obrazec za zahtevo za ogled — izpolnite ga (ali opišite svojo zahtevo s svojimi besedami in naj ga AI izpolni namesto vas).",
    },
    "phase_survey": {
        "English": "SITE SURVEY",
        "Italiano": "SOPRALLUOGO",
        "Deutsch": "BESICHTIGUNG",
        "Español": "VISITA TÉCNICA",
        "Slovenščina": "OGLED",
    },
    "phase_installation": {
        "English": "INSTALLATION",
        "Italiano": "INSTALLAZIONE",
        "Deutsch": "INSTALLATION",
        "Español": "INSTALACIÓN",
        "Slovenščina": "INSTALACIJA",
    },
    "ticket_type_door": {
        "English": "DOOR",
        "Italiano": "PORTA",
        "Deutsch": "TÜR",
        "Español": "PUERTA",
        "Slovenščina": "VRATA",
    },
    "ticket_type_window": {
        "English": "WINDOW",
        "Italiano": "FINESTRA",
        "Deutsch": "FENSTER",
        "Español": "VENTANA",
        "Slovenščina": "OKNO",
    },
    "map_tab_label": {
        "English": "Map",
        "Italiano": "Mappa",
        "Deutsch": "Karte",
        "Español": "Mapa",
        "Slovenščina": "Zemljevid",
    },
    "map_no_locations": {
        "English": "No tickets with a known location yet.",
        "Italiano": "Nessun ticket con una posizione nota al momento.",
        "Deutsch": "Noch keine Tickets mit bekanntem Standort.",
        "Español": "Aún no hay tickets con una ubicación conocida.",
        "Slovenščina": "Trenutno ni vstopnic z znano lokacijo.",
    },
    "map_unplaced_tickets_warning": {
        "English": "Could not place these tickets on the map (check/correct the address):",
        "Italiano": "Impossibile posizionare questi ticket sulla mappa (verifica/correggi l'indirizzo):",
        "Deutsch": "Diese Tickets konnten nicht auf der Karte platziert werden (Adresse prüfen/korrigieren):",
        "Español": "No se pudieron ubicar estos tickets en el mapa (revisa/corrige la dirección):",
        "Slovenščina": "Teh vstopnic ni bilo mogoče umestiti na zemljevid (preverite/popravite naslov):",
    },
    "map_no_address": {
        "English": "no address on file",
        "Italiano": "nessun indirizzo registrato",
        "Deutsch": "keine Adresse hinterlegt",
        "Español": "sin dirección registrada",
        "Slovenščina": "naslov ni na voljo",
    },
    "ticket_log_label": {
        "English": "History",
        "Italiano": "Cronologia",
        "Deutsch": "Verlauf",
        "Español": "Historial",
        "Slovenščina": "Zgodovina",
    },
    "briefing_estimated_time": {
        "English": "Estimated time",
        "Italiano": "Tempo stimato",
        "Deutsch": "Geschätzte Zeit",
        "Español": "Tiempo estimado",
        "Slovenščina": "Ocenjeni čas",
    },
    "briefing_on_site": {
        "English": "on-site",
        "Italiano": "in loco",
        "Deutsch": "vor Ort",
        "Español": "en el sitio",
        "Slovenščina": "na kraju",
    },
    "briefing_round_trip_travel": {
        "English": "round-trip travel",
        "Italiano": "viaggio andata/ritorno",
        "Deutsch": "Hin- und Rückfahrt",
        "Español": "viaje de ida y vuelta",
        "Slovenščina": "pot tja in nazaj",
    },
    "briefing_client_request": {
        "English": "From the client's request:",
        "Italiano": "Dalla richiesta del cliente:",
        "Deutsch": "Aus der Anfrage des Kunden:",
        "Español": "De la solicitud del cliente:",
        "Slovenščina": "Iz zahteve stranke:",
    },
    "briefing_surveyor_note": {
        "English": "Surveyor's note:",
        "Italiano": "Nota del sopralluogo:",
        "Deutsch": "Notiz der Besichtigung:",
        "Español": "Nota de la visita técnica:",
        "Slovenščina": "Opomba z ogleda:",
    },
    "briefing_installation_guidance": {
        "English": "Installation guidance for {model}",
        "Italiano": "Guida all'installazione per {model}",
        "Deutsch": "Installationsanleitung für {model}",
        "Español": "Guía de instalación para {model}",
        "Slovenščina": "Navodila za montažo {model}",
    },
    "briefing_install_query": {
        "English": "What are the general installation procedures for windows/doors like the {model} model?",
        "Italiano": "Quali sono le procedure generali di installazione per infissi (finestre/porte) come il modello {model}?",
        "Deutsch": "Was sind die allgemeinen Installationsverfahren für Fenster/Türen wie das Modell {model}?",
        "Español": "¿Cuáles son los procedimientos generales de instalación para ventanas/puertas como el modelo {model}?",
        "Slovenščina": "Kateri so splošni postopki montaže oken/vrat, kot je model {model}?",
    },
    "briefing_guidance_unavailable": {
        "English": "Installation guidance unavailable: {error}",
        "Italiano": "Guida all'installazione non disponibile: {error}",
        "Deutsch": "Installationsanleitung nicht verfügbar: {error}",
        "Español": "Guía de instalación no disponible: {error}",
        "Slovenščina": "Navodila za montažo niso na voljo: {error}",
    },
    "briefing_no_specific_guidance": {
        "English": "No installation guide specific to {model} was found. Here is the general guidance that applies: ",
        "Italiano": "Non è stata trovata una guida di installazione specifica per {model}. Ecco le indicazioni generali applicabili: ",
        "Deutsch": "Es wurde keine spezifische Installationsanleitung für {model} gefunden. Hier sind die allgemeinen Hinweise, die gelten: ",
        "Español": "No se encontró una guía de instalación específica para {model}. Aquí tienes las indicaciones generales que aplican: ",
        "Slovenščina": "Ni bilo najdenih posebnih navodil za montažo za {model}. Tukaj so splošna navodila, ki veljajo: ",
    },
    # Ticket status labels (badges, table Status column, filter dropdown)
    "status_new": {
        "English": "New",
        "Italiano": "Nuovo",
        "Deutsch": "Neu",
        "Español": "Nuevo",
        "Slovenščina": "Novo",
    },
    "status_open": {
        "English": "Open",
        "Italiano": "Aperto",
        "Deutsch": "Offen",
        "Español": "Abierto",
        "Slovenščina": "Odprto",
    },
    "status_in_progress": {
        "English": "In Progress",
        "Italiano": "In corso",
        "Deutsch": "In Bearbeitung",
        "Español": "En curso",
        "Slovenščina": "V teku",
    },
    "status_awaiting_client": {
        "English": "Awaiting Client",
        "Italiano": "In attesa del cliente",
        "Deutsch": "Wartet auf Kunde",
        "Español": "Esperando al cliente",
        "Slovenščina": "Čaka na stranko",
    },
    "status_blocked": {
        "English": "Blocked",
        "Italiano": "Bloccato",
        "Deutsch": "Blockiert",
        "Español": "Bloqueado",
        "Slovenščina": "Blokirano",
    },
    "status_in_review": {
        "English": "In Review",
        "Italiano": "In revisione",
        "Deutsch": "In Prüfung",
        "Español": "En revisión",
        "Slovenščina": "V pregledu",
    },
    "status_closed": {
        "English": "Closed",
        "Italiano": "Chiuso",
        "Deutsch": "Geschlossen",
        "Español": "Cerrado",
        "Slovenščina": "Zaprto",
    },
    "status_cancelled": {
        "English": "Cancelled",
        "Italiano": "Annullato",
        "Deutsch": "Storniert",
        "Español": "Cancelado",
        "Slovenščina": "Preklicano",
    },
    # Auth gate
    "please_login": {
        "English": "Please log in from the main page.",
        "Italiano": "Effettua l'accesso dalla pagina principale.",
        "Deutsch": "Bitte melden Sie sich über die Hauptseite an.",
        "Español": "Inicia sesión desde la página principal.",
        "Slovenščina": "Prijavite se na glavni strani.",
    },
    "access_denied": {
        "English": "Access denied. This page is for: {roles}.",
        "Italiano": "Accesso negato. Questa pagina è per: {roles}.",
        "Deutsch": "Zugriff verweigert. Diese Seite ist für: {roles}.",
        "Español": "Acceso denegado. Esta página es para: {roles}.",
        "Slovenščina": "Dostop zavrnjen. Ta stran je namenjena: {roles}.",
    },
    # Sidebar extras
    "active_llm_label": {
        "English": "Active LLM",
        "Italiano": "LLM attivo",
        "Deutsch": "Aktives LLM",
        "Español": "LLM activo",
        "Slovenščina": "Aktivni LLM",
    },
    "catalogue_select_label": {
        "English": "Catalogue",
        "Italiano": "Catalogo",
        "Deutsch": "Katalog",
        "Español": "Catálogo",
        "Slovenščina": "Katalog",
    },
    # Shared ticket card / list bits
    "no_tickets_yet": {
        "English": "No tickets yet.",
        "Italiano": "Nessun ticket ancora.",
        "Deutsch": "Noch keine Tickets.",
        "Español": "Aún no hay tickets.",
        "Slovenščina": "Še ni vstopnic.",
    },
    "stock_not_found": {
        "English": "Stock: model not found",
        "Italiano": "Scorte: modello non trovato",
        "Deutsch": "Bestand: Modell nicht gefunden",
        "Español": "Stock: modelo no encontrado",
        "Slovenščina": "Zaloga: model ni najden",
    },
    "address_inline": {
        "English": "Address: {address}",
        "Italiano": "Indirizzo: {address}",
        "Deutsch": "Adresse: {address}",
        "Español": "Dirección: {address}",
        "Slovenščina": "Naslov: {address}",
    },
    "notes_inline": {
        "English": "Notes: {notes}",
        "Italiano": "Note: {notes}",
        "Deutsch": "Notizen: {notes}",
        "Español": "Notas: {notes}",
        "Slovenščina": "Opombe: {notes}",
    },
    "installer_inline": {
        "English": "Installer: {installer} — started {date}",
        "Italiano": "Installatore: {installer} — iniziato il {date}",
        "Deutsch": "Monteur: {installer} — begonnen am {date}",
        "Español": "Instalador: {installer} — iniciado el {date}",
        "Slovenščina": "Monter: {installer} — začeto {date}",
    },
    # HITL / voice / turn handling
    "confirm_cancel_hint": {
        "English": 'Or type / say "confirm" or "cancel".',
        "Italiano": 'Oppure scrivi o pronuncia "conferma" o "annulla".',
        "Deutsch": 'Oder tippen/sagen Sie "bestätigen" oder "abbrechen".',
        "Español": 'O escribe o di "confirmar" o "cancelar".',
        "Slovenščina": 'Ali natipkajte oz. recite "potrdi" ali "prekliči".',
    },
    "no_speech_detected": {
        "English": "No speech detected. Please try again.",
        "Italiano": "Nessun parlato rilevato. Riprova.",
        "Deutsch": "Keine Sprache erkannt. Bitte erneut versuchen.",
        "Español": "No se detectó voz. Inténtalo de nuevo.",
        "Slovenščina": "Govor ni bil zaznan. Poskusite znova.",
    },
    "transcription_failed": {
        "English": "Transcription failed: {error}",
        "Italiano": "Trascrizione non riuscita: {error}",
        "Deutsch": "Transkription fehlgeschlagen: {error}",
        "Español": "Error en la transcripción: {error}",
        "Slovenščina": "Prepis ni uspel: {error}",
    },
    "hitl_pending_prompt": {
        "English": 'Please **confirm** or **cancel** the pending action first.  \nSay or type "confirm" / "cancel" (or use the buttons above).',
        "Italiano": 'Per favore **conferma** o **annulla** prima l\'azione in sospeso.  \nDì o scrivi "conferma" / "annulla" (oppure usa i pulsanti sopra).',
        "Deutsch": 'Bitte **bestätigen** oder **brechen Sie ab** zuerst die ausstehende Aktion.  \nSagen oder tippen Sie "bestätigen" / "abbrechen" (oder nutzen Sie die Schaltflächen oben).',
        "Español": 'Por favor **confirma** o **cancela** primero la acción pendiente.  \nDi o escribe "confirmar" / "cancelar" (o usa los botones de arriba).',
        "Slovenščina": 'Prosimo, najprej **potrdite** ali **prekličite** čakajoče dejanje.  \nRecite ali natipkajte "potrdi" / "prekliči" (ali uporabite zgornje gumbe).',
    },
    "session_limit_reached": {
        "English": "Session limit reached.",
        "Italiano": "Limite di sessione raggiunto.",
        "Deutsch": "Sitzungslimit erreicht.",
        "Español": "Se alcanzó el límite de la sesión.",
        "Slovenščina": "Dosežena je omejitev seje.",
    },
    "please_wait_between_messages": {
        "English": "Please wait a moment between messages.",
        "Italiano": "Attendi un momento tra un messaggio e l'altro.",
        "Deutsch": "Bitte warten Sie einen Moment zwischen den Nachrichten.",
        "Español": "Espera un momento entre mensajes.",
        "Slovenščina": "Prosimo, počakajte trenutek med sporočili.",
    },
    # Manager — reset dialog
    "reset_dialog_title": {
        "English": "Reset all operational data?",
        "Italiano": "Ripristinare tutti i dati operativi?",
        "Deutsch": "Alle Betriebsdaten zurücksetzen?",
        "Español": "¿Restablecer todos los datos operativos?",
        "Slovenščina": "Ponastavi vse operativne podatke?",
    },
    "reset_dialog_warning": {
        "English": "This will permanently delete **all tickets, alerts, job reports, and pending QA entries**. Inventory, users, clients, and cost settings are preserved. This cannot be undone.",
        "Italiano": "Questa azione eliminerà definitivamente **tutti i ticket, gli avvisi, i rapporti di lavoro e le voci QA in sospeso**. Inventario, utenti, clienti e impostazioni dei costi vengono conservati. L'operazione non è reversibile.",
        "Deutsch": "Dadurch werden **alle Tickets, Warnungen, Auftragsberichte und ausstehenden QA-Einträge** dauerhaft gelöscht. Bestand, Benutzer, Kunden und Kosteneinstellungen bleiben erhalten. Dies kann nicht rückgängig gemacht werden.",
        "Español": "Esto eliminará permanentemente **todos los tickets, alertas, informes de trabajo y entradas de control de calidad pendientes**. El inventario, los usuarios, los clientes y la configuración de costos se conservan. Esta acción no se puede deshacer.",
        "Slovenščina": "S tem boste trajno izbrisali **vse vstopnice, opozorila, poročila o delu in čakajoče vnose nadzora kakovosti**. Zaloga, uporabniki, stranke in nastavitve stroškov ostanejo ohranjeni. Tega dejanja ni mogoče razveljaviti.",
    },
    "reset_confirm_button": {
        "English": "Yes, reset everything",
        "Italiano": "Sì, ripristina tutto",
        "Deutsch": "Ja, alles zurücksetzen",
        "Español": "Sí, restablecer todo",
        "Slovenščina": "Da, ponastavi vse",
    },
    "reset_success": {
        "English": "All operational data deleted.",
        "Italiano": "Tutti i dati operativi sono stati eliminati.",
        "Deutsch": "Alle Betriebsdaten wurden gelöscht.",
        "Español": "Se eliminaron todos los datos operativos.",
        "Slovenščina": "Vsi operativni podatki so bili izbrisani.",
    },
    # Manager — dashboard tabs
    "tab_tickets": {
        "English": "Tickets",
        "Italiano": "Ticket",
        "Deutsch": "Tickets",
        "Español": "Tickets",
        "Slovenščina": "Vstopnice",
    },
    "tab_clients": {
        "English": "Clients",
        "Italiano": "Clienti",
        "Deutsch": "Kunden",
        "Español": "Clientes",
        "Slovenščina": "Stranke",
    },
    "tab_alerts": {
        "English": "Alerts",
        "Italiano": "Avvisi",
        "Deutsch": "Warnungen",
        "Español": "Alertas",
        "Slovenščina": "Opozorila",
    },
    "tab_qa": {
        "English": "Knowledge Cache",
        "Italiano": "Cache di conoscenza",
        "Deutsch": "Wissenscache",
        "Español": "Caché de conocimiento",
        "Slovenščina": "Predpomnilnik znanja",
    },
    "tab_analytics": {
        "English": "Analytics",
        "Italiano": "Analisi",
        "Deutsch": "Analysen",
        "Español": "Analítica",
        "Slovenščina": "Analitika",
    },
    "tab_llm_usage": {
        "English": "LLM Usage",
        "Italiano": "Utilizzo LLM",
        "Deutsch": "LLM-Nutzung",
        "Español": "Uso de LLM",
        "Slovenščina": "Uporaba LLM",
    },
    "tab_my_jobs": {
        "English": "My Jobs",
        "Italiano": "I miei lavori",
        "Deutsch": "Meine Aufträge",
        "Español": "Mis trabajos",
        "Slovenščina": "Moji posli",
    },
    "tab_available": {
        "English": "Available",
        "Italiano": "Disponibili",
        "Deutsch": "Verfügbar",
        "Español": "Disponibles",
        "Slovenščina": "Na voljo",
    },
    # Manager — ticket list / filter
    "ticket_filter_label": {
        "English": "Filter",
        "Italiano": "Filtro",
        "Deutsch": "Filter",
        "Español": "Filtro",
        "Slovenščina": "Filter",
    },
    "ticket_type_filter_label": {
        "English": "Type",
        "Italiano": "Tipo",
        "Deutsch": "Typ",
        "Español": "Tipo",
        "Slovenščina": "Vrsta",
    },
    "filter_all": {
        "English": "All",
        "Italiano": "Tutti",
        "Deutsch": "Alle",
        "Español": "Todos",
        "Slovenščina": "Vse",
    },
    "approve_all_new": {
        "English": "Approve all NEW ({count})",
        "Italiano": "Approva tutti i NUOVI ({count})",
        "Deutsch": "Alle NEUEN genehmigen ({count})",
        "Español": "Aprobar todos los NUEVOS ({count})",
        "Slovenščina": "Odobri vse NOVE ({count})",
    },
    "approved_n_tickets": {
        "English": "Approved {count} ticket(s).",
        "Italiano": "Approvati {count} ticket.",
        "Deutsch": "{count} Ticket(s) genehmigt.",
        "Español": "Se aprobaron {count} ticket(s).",
        "Slovenščina": "Odobrenih {count} vstopnic.",
    },
    "no_status_tickets": {
        "English": "No {status} tickets.",
        "Italiano": "Nessun ticket {status}.",
        "Deutsch": "Keine Tickets mit Status {status}.",
        "Español": "No hay tickets {status}.",
        "Slovenščina": "Ni vstopnic s statusom {status}.",
    },
    "select_ticket_hint": {
        "English": "Select a ticket row above to see its details and actions.",
        "Italiano": "Seleziona una riga del ticket qui sopra per vederne i dettagli e le azioni.",
        "Deutsch": "Wählen Sie oben eine Ticketzeile aus, um Details und Aktionen zu sehen.",
        "Español": "Selecciona una fila de ticket arriba para ver sus detalles y acciones.",
        "Slovenščina": "Zgoraj izberite vrstico vstopnice, da vidite podrobnosti in dejanja.",
    },
    "include_in_batch": {
        "English": "Include in batch approve",
        "Italiano": "Includi nell'approvazione multipla",
        "Deutsch": "In Sammelgenehmigung einbeziehen",
        "Español": "Incluir en aprobación por lotes",
        "Slovenščina": "Vključi v skupno odobritev",
    },
    "approve_button": {
        "English": "Approve",
        "Italiano": "Approva",
        "Deutsch": "Genehmigen",
        "Español": "Aprobar",
        "Slovenščina": "Odobri",
    },
    "reject_button": {
        "English": "Reject",
        "Italiano": "Rifiuta",
        "Deutsch": "Ablehnen",
        "Español": "Rechazar",
        "Slovenščina": "Zavrni",
    },
    "reason_for_rejection": {
        "English": "Reason for rejection",
        "Italiano": "Motivo del rifiuto",
        "Deutsch": "Grund für die Ablehnung",
        "Español": "Motivo del rechazo",
        "Slovenščina": "Razlog za zavrnitev",
    },
    "reason_rejection_placeholder": {
        "English": "e.g. Slot unavailable — please contact us to reschedule.",
        "Italiano": "es. Fascia oraria non disponibile — contattaci per riprogrammare.",
        "Deutsch": "z. B. Termin nicht verfügbar — bitte kontaktieren Sie uns zur Umplanung.",
        "Español": "p. ej. Horario no disponible — contáctanos para reprogramar.",
        "Slovenščina": "npr. Termin ni na voljo — kontaktirajte nas za prestavitev.",
    },
    "confirm_rejection_button": {
        "English": "Confirm rejection",
        "Italiano": "Conferma rifiuto",
        "Deutsch": "Ablehnung bestätigen",
        "Español": "Confirmar rechazo",
        "Slovenščina": "Potrdi zavrnitev",
    },
    "please_provide_reason": {
        "English": "Please provide a reason before rejecting.",
        "Italiano": "Fornisci un motivo prima di rifiutare.",
        "Deutsch": "Bitte geben Sie vor der Ablehnung einen Grund an.",
        "Español": "Proporciona un motivo antes de rechazar.",
        "Slovenščina": "Pred zavrnitvijo navedite razlog.",
    },
    "view_alert_expander": {
        "English": "View alert",
        "Italiano": "Visualizza avviso",
        "Deutsch": "Warnung anzeigen",
        "Español": "Ver alerta",
        "Slovenščina": "Prikaži opozorilo",
    },
    "view_survey_measurements": {
        "English": "View survey measurements",
        "Italiano": "Visualizza misurazioni del sopralluogo",
        "Deutsch": "Vermessungsdaten anzeigen",
        "Español": "Ver mediciones de la visita técnica",
        "Slovenščina": "Prikaži meritve ogleda",
    },
    "dimensions_label": {
        "English": "Dimensions:",
        "Italiano": "Dimensioni:",
        "Deutsch": "Abmessungen:",
        "Español": "Dimensiones:",
        "Slovenščina": "Dimenzije:",
    },
    "glass_type_caption": {
        "English": "Glass type:",
        "Italiano": "Tipo di vetro:",
        "Deutsch": "Glasart:",
        "Español": "Tipo de vidrio:",
        "Slovenščina": "Vrsta stekla:",
    },
    "approve_and_price": {
        "English": "Approve & Price",
        "Italiano": "Approva e quota",
        "Deutsch": "Genehmigen & Bepreisen",
        "Español": "Aprobar y cotizar",
        "Slovenščina": "Odobri in oceni ceno",
    },
    "view_completion_report": {
        "English": "View completion report",
        "Italiano": "Visualizza rapporto di completamento",
        "Deutsch": "Abschlussbericht anzeigen",
        "Español": "Ver informe de finalización",
        "Slovenščina": "Prikaži poročilo o zaključku",
    },
    "client_email_thankyou_label": {
        "English": "Client email for thank-you",
        "Italiano": "Email cliente per il ringraziamento",
        "Deutsch": "Kunden-E-Mail für Dankesschreiben",
        "Español": "Correo del cliente para el agradecimiento",
        "Slovenščina": "E-pošta stranke za zahvalo",
    },
    "approve_and_email": {
        "English": "Approve & Email",
        "Italiano": "Approva e invia email",
        "Deutsch": "Genehmigen & E-Mail senden",
        "Español": "Aprobar y enviar correo",
        "Slovenščina": "Odobri in pošlji e-pošto",
    },
    "email_sent_msg": {
        "English": "Approved. Email {status} to {email}.",
        "Italiano": "Approvato. Email {status} a {email}.",
        "Deutsch": "Genehmigt. E-Mail {status} an {email}.",
        "Español": "Aprobado. Correo {status} a {email}.",
        "Slovenščina": "Odobreno. E-pošta {status} na {email}.",
    },
    "email_status_sent": {
        "English": "sent",
        "Italiano": "inviata",
        "Deutsch": "gesendet",
        "Español": "enviado",
        "Slovenščina": "poslano",
    },
    "email_status_failed": {
        "English": "failed (SMTP)",
        "Italiano": "non riuscita (SMTP)",
        "Deutsch": "fehlgeschlagen (SMTP)",
        "Español": "fallido (SMTP)",
        "Slovenščina": "neuspešno (SMTP)",
    },
    "email_error_msg": {
        "English": "Approved, email error: {error}",
        "Italiano": "Approvato, errore email: {error}",
        "Deutsch": "Genehmigt, E-Mail-Fehler: {error}",
        "Español": "Aprobado, error de correo: {error}",
        "Slovenščina": "Odobreno, napaka e-pošte: {error}",
    },
    "approved_no_email_msg": {
        "English": "Approved. No client email — skipped email.",
        "Italiano": "Approvato. Nessuna email cliente — invio email saltato.",
        "Deutsch": "Genehmigt. Keine Kunden-E-Mail — E-Mail übersprungen.",
        "Español": "Aprobado. Sin correo del cliente — se omitió el envío.",
        "Slovenščina": "Odobreno. Ni e-pošte stranke — e-pošta preskočena.",
    },
    "report_rejected_msg": {
        "English": "Report rejected — ticket back to IN PROGRESS.",
        "Italiano": "Rapporto rifiutato — ticket tornato IN CORSO.",
        "Deutsch": "Bericht abgelehnt — Ticket zurück auf IN BEARBEITUNG.",
        "Español": "Informe rechazado — ticket vuelve a EN CURSO.",
        "Slovenščina": "Poročilo zavrnjeno — vstopnica nazaj V TEKU.",
    },
    "estimate_sent_caption": {
        "English": "Estimate sent: €{price} — awaiting the client's response.",
        "Italiano": "Preventivo inviato: €{price} — in attesa della risposta del cliente.",
        "Deutsch": "Angebot gesendet: €{price} — wartet auf Antwort des Kunden.",
        "Español": "Presupuesto enviado: €{price} — esperando la respuesta del cliente.",
        "Slovenščina": "Ponudba poslana: €{price} — čaka se odgovor stranke.",
    },
    "edit_ticket_expander": {
        "English": "Edit ticket",
        "Italiano": "Modifica ticket",
        "Deutsch": "Ticket bearbeiten",
        "Español": "Editar ticket",
        "Slovenščina": "Uredi vstopnico",
    },
    "client_name_label": {
        "English": "Client name",
        "Italiano": "Nome cliente",
        "Deutsch": "Kundenname",
        "Español": "Nombre del cliente",
        "Slovenščina": "Ime stranke",
    },
    "client_email_label": {
        "English": "Client email",
        "Italiano": "Email cliente",
        "Deutsch": "Kunden-E-Mail",
        "Español": "Correo del cliente",
        "Slovenščina": "E-pošta stranke",
    },
    "client_phone_label": {
        "English": "Client phone",
        "Italiano": "Telefono cliente",
        "Deutsch": "Kundentelefon",
        "Español": "Teléfono del cliente",
        "Slovenščina": "Telefon stranke",
    },
    "client_address_label": {
        "English": "Client address",
        "Italiano": "Indirizzo cliente",
        "Deutsch": "Kundenadresse",
        "Español": "Dirección del cliente",
        "Slovenščina": "Naslov stranke",
    },
    "notes_field_label": {
        "English": "Notes",
        "Italiano": "Note",
        "Deutsch": "Notizen",
        "Español": "Notas",
        "Slovenščina": "Opombe",
    },
    "save_changes_button": {
        "English": "Save changes",
        "Italiano": "Salva modifiche",
        "Deutsch": "Änderungen speichern",
        "Español": "Guardar cambios",
        "Slovenščina": "Shrani spremembe",
    },
    "date_time_format_error": {
        "English": "Date must be YYYY-MM-DD and time must be HH:MM.",
        "Italiano": "La data deve essere AAAA-MM-GG e l'ora HH:MM.",
        "Deutsch": "Datum muss JJJJ-MM-TT und Uhrzeit HH:MM sein.",
        "Español": "La fecha debe ser AAAA-MM-DD y la hora HH:MM.",
        "Slovenščina": "Datum mora biti LLLL-MM-DD, ura pa UU:MM.",
    },
    "ticket_updated_msg": {
        "English": "{code} updated.",
        "Italiano": "{code} aggiornato.",
        "Deutsch": "{code} aktualisiert.",
        "Español": "{code} actualizado.",
        "Slovenščina": "{code} posodobljena.",
    },
    "could_not_update_ticket": {
        "English": "Could not update this ticket.",
        "Italiano": "Impossibile aggiornare questo ticket.",
        "Deutsch": "Dieses Ticket konnte nicht aktualisiert werden.",
        "Español": "No se pudo actualizar este ticket.",
        "Slovenščina": "Te vstopnice ni bilo mogoče posodobiti.",
    },
    # Manager — clients tab
    "search_label": {
        "English": "Search",
        "Italiano": "Cerca",
        "Deutsch": "Suchen",
        "Español": "Buscar",
        "Slovenščina": "Iskanje",
    },
    "search_by_name_placeholder": {
        "English": "Search by name...",
        "Italiano": "Cerca per nome...",
        "Deutsch": "Nach Namen suchen...",
        "Español": "Buscar por nombre...",
        "Slovenščina": "Iskanje po imenu...",
    },
    "no_clients_yet": {
        "English": "No clients yet.",
        "Italiano": "Nessun cliente ancora.",
        "Deutsch": "Noch keine Kunden.",
        "Español": "Aún no hay clientes.",
        "Slovenščina": "Še ni strank.",
    },
    "no_clients_match": {
        "English": "No clients match your search.",
        "Italiano": "Nessun cliente corrisponde alla ricerca.",
        "Deutsch": "Keine Kunden entsprechen der Suche.",
        "Español": "Ningún cliente coincide con tu búsqueda.",
        "Slovenščina": "Nobena stranka ne ustreza iskanju.",
    },
    "email_label_bold": {
        "English": "Email:",
        "Italiano": "Email:",
        "Deutsch": "E-Mail:",
        "Español": "Correo:",
        "Slovenščina": "E-pošta:",
    },
    "phone_label_inline": {
        "English": "Phone: {phone}",
        "Italiano": "Telefono: {phone}",
        "Deutsch": "Telefon: {phone}",
        "Español": "Teléfono: {phone}",
        "Slovenščina": "Telefon: {phone}",
    },
    "client_since_label": {
        "English": "Client since: {date}",
        "Italiano": "Cliente dal: {date}",
        "Deutsch": "Kunde seit: {date}",
        "Español": "Cliente desde: {date}",
        "Slovenščina": "Stranka od: {date}",
    },
    # Manager — alerts tab
    "no_alerts_yet": {
        "English": "No alerts from installers yet.",
        "Italiano": "Nessun avviso dagli installatori.",
        "Deutsch": "Noch keine Warnungen von Monteuren.",
        "Español": "Aún no hay alertas de los instaladores.",
        "Slovenščina": "Še ni opozoril monterjev.",
    },
    "ticket_ref_inline": {
        "English": "Ticket {code}",
        "Italiano": "Ticket {code}",
        "Deutsch": "Ticket {code}",
        "Español": "Ticket {code}",
        "Slovenščina": "Vstopnica {code}",
    },
    "ref_label_inline": {
        "English": "Ref: {ref}",
        "Italiano": "Rif: {ref}",
        "Deutsch": "Ref: {ref}",
        "Español": "Ref: {ref}",
        "Slovenščina": "Ref: {ref}",
    },
    "mark_all_read_button": {
        "English": "Mark all as read",
        "Italiano": "Segna tutti come letti",
        "Deutsch": "Alle als gelesen markieren",
        "Español": "Marcar todo como leído",
        "Slovenščina": "Označi vse kot prebrano",
    },
    "all_alerts_expander": {
        "English": "All alerts ({count})",
        "Italiano": "Tutti gli avvisi ({count})",
        "Deutsch": "Alle Warnungen ({count})",
        "Español": "Todas las alertas ({count})",
        "Slovenščina": "Vsa opozorila ({count})",
    },
    "read_label": {
        "English": "read",
        "Italiano": "letto",
        "Deutsch": "gelesen",
        "Español": "leído",
        "Slovenščina": "prebrano",
    },
    # Manager — QA cache tab
    "no_qa_pending": {
        "English": "No Q&A pairs pending cache review.",
        "Italiano": "Nessuna coppia D/R in attesa di revisione.",
        "Deutsch": "Keine Frage-Antwort-Paare zur Überprüfung ausstehend.",
        "Español": "No hay pares de preguntas y respuestas pendientes de revisión.",
        "Slovenščina": "Ni parov vprašanje-odgovor, ki čakajo na pregled.",
    },
    "qa_question_label": {
        "English": "Q ({role}):",
        "Italiano": "D ({role}):",
        "Deutsch": "F ({role}):",
        "Español": "P ({role}):",
        "Slovenščina": "V ({role}):",
    },
    "submitted_label": {
        "English": "Submitted: {date}",
        "Italiano": "Inviato: {date}",
        "Deutsch": "Eingereicht: {date}",
        "Español": "Enviado: {date}",
        "Slovenščina": "Oddano: {date}",
    },
    "view_answer_expander": {
        "English": "View answer",
        "Italiano": "Visualizza risposta",
        "Deutsch": "Antwort anzeigen",
        "Español": "Ver respuesta",
        "Slovenščina": "Prikaži odgovor",
    },
    "add_to_cache_button": {
        "English": "Add to cache",
        "Italiano": "Aggiungi alla cache",
        "Deutsch": "Zum Cache hinzufügen",
        "Español": "Añadir a la caché",
        "Slovenščina": "Dodaj v predpomnilnik",
    },
    "added_msg": {
        "English": "Added.",
        "Italiano": "Aggiunto.",
        "Deutsch": "Hinzugefügt.",
        "Español": "Añadido.",
        "Slovenščina": "Dodano.",
    },
    "skipped_duplicate_msg": {
        "English": "Skipped — near-identical entry exists.",
        "Italiano": "Saltato — esiste già una voce quasi identica.",
        "Deutsch": "Übersprungen — nahezu identischer Eintrag vorhanden.",
        "Español": "Omitido — ya existe una entrada casi idéntica.",
        "Slovenščina": "Preskočeno — obstaja skoraj enak vnos.",
    },
    "skip_button": {
        "English": "Skip",
        "Italiano": "Salta",
        "Deutsch": "Überspringen",
        "Español": "Omitir",
        "Slovenščina": "Preskoči",
    },
    # Manager — analytics / usage tabs
    "period_label": {
        "English": "Period",
        "Italiano": "Periodo",
        "Deutsch": "Zeitraum",
        "Español": "Período",
        "Slovenščina": "Obdobje",
    },
    "period_day": {
        "English": "Day",
        "Italiano": "Giorno",
        "Deutsch": "Tag",
        "Español": "Día",
        "Slovenščina": "Dan",
    },
    "period_week": {
        "English": "Week",
        "Italiano": "Settimana",
        "Deutsch": "Woche",
        "Español": "Semana",
        "Slovenščina": "Teden",
    },
    "period_month": {
        "English": "Month",
        "Italiano": "Mese",
        "Deutsch": "Monat",
        "Español": "Mes",
        "Slovenščina": "Mesec",
    },
    "closed_tickets_label": {
        "English": "Closed Tickets",
        "Italiano": "Ticket chiusi",
        "Deutsch": "Geschlossene Tickets",
        "Español": "Tickets cerrados",
        "Slovenščina": "Zaprte vstopnice",
    },
    "total_revenue_label": {
        "English": "Total Revenue",
        "Italiano": "Ricavo totale",
        "Deutsch": "Gesamtumsatz",
        "Español": "Ingresos totales",
        "Slovenščina": "Skupni prihodki",
    },
    "hours_worked_label": {
        "English": "Hours Worked",
        "Italiano": "Ore lavorate",
        "Deutsch": "Gearbeitete Stunden",
        "Español": "Horas trabajadas",
        "Slovenščina": "Opravljene ure",
    },
    "no_closed_tickets_period": {
        "English": "No closed tickets in this period yet.",
        "Italiano": "Nessun ticket chiuso in questo periodo.",
        "Deutsch": "Noch keine geschlossenen Tickets in diesem Zeitraum.",
        "Español": "Aún no hay tickets cerrados en este período.",
        "Slovenščina": "V tem obdobju še ni zaprtih vstopnic.",
    },
    "cost_label": {
        "English": "Cost",
        "Italiano": "Costo",
        "Deutsch": "Kosten",
        "Español": "Costo",
        "Slovenščina": "Strošek",
    },
    "requests_label": {
        "English": "Requests",
        "Italiano": "Richieste",
        "Deutsch": "Anfragen",
        "Español": "Solicitudes",
        "Slovenščina": "Zahteve",
    },
    "tokens_label": {
        "English": "Tokens",
        "Italiano": "Token",
        "Deutsch": "Tokens",
        "Español": "Tokens",
        "Slovenščina": "Žetoni",
    },
    "no_usage_period": {
        "English": "No usage recorded for this period yet.",
        "Italiano": "Nessun utilizzo registrato in questo periodo.",
        "Deutsch": "Noch keine Nutzung in diesem Zeitraum erfasst.",
        "Español": "Aún no se registró uso en este período.",
        "Slovenščina": "V tem obdobju še ni zabeležene uporabe.",
    },
    "reset_all_button": {
        "English": "Reset all tickets & dashboard data",
        "Italiano": "Ripristina tutti i ticket e i dati della dashboard",
        "Deutsch": "Alle Tickets & Dashboard-Daten zurücksetzen",
        "Español": "Restablecer todos los tickets y datos del panel",
        "Slovenščina": "Ponastavi vse vstopnice in podatke nadzorne plošče",
    },
    # Installer — job actions
    "release_button": {
        "English": "Release",
        "Italiano": "Rilascia",
        "Deutsch": "Freigeben",
        "Español": "Liberar",
        "Slovenščina": "Sprosti",
    },
    "released_msg": {
        "English": "{code} released.",
        "Italiano": "{code} rilasciato.",
        "Deutsch": "{code} freigegeben.",
        "Español": "{code} liberado.",
        "Slovenščina": "{code} sproščena.",
    },
    "could_not_release": {
        "English": "Could not release this job.",
        "Italiano": "Impossibile rilasciare questo lavoro.",
        "Deutsch": "Dieser Auftrag konnte nicht freigegeben werden.",
        "Español": "No se pudo liberar este trabajo.",
        "Slovenščina": "Tega posla ni bilo mogoče sprostiti.",
    },
    "report_measurements_button": {
        "English": "Report Measurements",
        "Italiano": "Registra misurazioni",
        "Deutsch": "Messungen melden",
        "Español": "Reportar mediciones",
        "Slovenščina": "Poroči o meritvah",
    },
    "close_report_button": {
        "English": "Close / Report",
        "Italiano": "Chiudi / Rapporto",
        "Deutsch": "Abschließen / Bericht",
        "Español": "Cerrar / Informe",
        "Slovenščina": "Zapri / Poročilo",
    },
    "height_cm_label": {
        "English": "Height (cm)",
        "Italiano": "Altezza (cm)",
        "Deutsch": "Höhe (cm)",
        "Español": "Altura (cm)",
        "Slovenščina": "Višina (cm)",
    },
    "width_cm_label": {
        "English": "Width (cm)",
        "Italiano": "Larghezza (cm)",
        "Deutsch": "Breite (cm)",
        "Español": "Ancho (cm)",
        "Slovenščina": "Širina (cm)",
    },
    "depth_cm_label": {
        "English": "Depth (cm)",
        "Italiano": "Profondità (cm)",
        "Deutsch": "Tiefe (cm)",
        "Español": "Profundidad (cm)",
        "Slovenščina": "Globina (cm)",
    },
    "glass_standard": {
        "English": "Standard (double low-E)",
        "Italiano": "Standard (doppio basso emissivo)",
        "Deutsch": "Standard (Doppel-Low-E)",
        "Español": "Estándar (doble bajo emisivo)",
        "Slovenščina": "Standardno (dvojno low-E)",
    },
    "glass_triple": {
        "English": "Triple glazing",
        "Italiano": "Triplo vetro",
        "Deutsch": "Dreifachverglasung",
        "Español": "Triple acristalamiento",
        "Slovenščina": "Trojna zasteklitev",
    },
    "glass_security": {
        "English": "Security (anti-break-in)",
        "Italiano": "Sicurezza (antieffrazione)",
        "Deutsch": "Sicherheit (einbruchhemmend)",
        "Español": "Seguridad (antirrobo)",
        "Slovenščina": "Varnostno (proti vlomu)",
    },
    "notes_optional_label": {
        "English": "Notes (optional)",
        "Italiano": "Note (opzionale)",
        "Deutsch": "Notizen (optional)",
        "Español": "Notas (opcional)",
        "Slovenščina": "Opombe (neobvezno)",
    },
    "notes_optional_placeholder": {
        "English": "Anything the manager should know...",
        "Italiano": "Qualsiasi cosa il manager dovrebbe sapere...",
        "Deutsch": "Alles, was der Manager wissen sollte...",
        "Español": "Cualquier cosa que el gerente deba saber...",
        "Slovenščina": "Karkoli bi moral vodja vedeti...",
    },
    "submit_measurements_button": {
        "English": "Submit Measurements",
        "Italiano": "Invia misurazioni",
        "Deutsch": "Messungen einreichen",
        "Español": "Enviar mediciones",
        "Slovenščina": "Pošlji meritve",
    },
    "height_width_depth_error": {
        "English": "Height, width, and depth must all be greater than zero.",
        "Italiano": "Altezza, larghezza e profondità devono essere tutte maggiori di zero.",
        "Deutsch": "Höhe, Breite und Tiefe müssen alle größer als null sein.",
        "Español": "La altura, el ancho y la profundidad deben ser mayores que cero.",
        "Slovenščina": "Višina, širina in globina morajo biti večje od nič.",
    },
    "completion_notes_label": {
        "English": "Completion notes",
        "Italiano": "Note di completamento",
        "Deutsch": "Abschlussnotizen",
        "Español": "Notas de finalización",
        "Slovenščina": "Opombe ob zaključku",
    },
    "completion_notes_placeholder": {
        "English": "What was done, any issues, outcome...",
        "Italiano": "Cosa è stato fatto, eventuali problemi, esito...",
        "Deutsch": "Was wurde getan, etwaige Probleme, Ergebnis...",
        "Español": "Qué se hizo, problemas, resultado...",
        "Slovenščina": "Kaj je bilo opravljeno, morebitne težave, izid...",
    },
    "submit_report_button": {
        "English": "Submit Report",
        "Italiano": "Invia rapporto",
        "Deutsch": "Bericht einreichen",
        "Español": "Enviar informe",
        "Slovenščina": "Pošlji poročilo",
    },
    "please_add_notes": {
        "English": "Please add some completion notes.",
        "Italiano": "Aggiungi alcune note di completamento.",
        "Deutsch": "Bitte fügen Sie Abschlussnotizen hinzu.",
        "Español": "Agrega notas de finalización.",
        "Slovenščina": "Dodajte opombe ob zaključku.",
    },
    "no_jobs_available": {
        "English": "No jobs available to claim.",
        "Italiano": "Nessun lavoro disponibile da rivendicare.",
        "Deutsch": "Keine Aufträge zum Beanspruchen verfügbar.",
        "Español": "No hay trabajos disponibles para reclamar.",
        "Slovenščina": "Ni razpoložljivih poslov za prevzem.",
    },
    "already_active_job_warning": {
        "English": "You already have an active job ({code} — {model} on {date}). Release it before claiming a new one.",
        "Italiano": "Hai già un lavoro attivo ({code} — {model} il {date}). Rilascialo prima di rivendicarne uno nuovo.",
        "Deutsch": "Sie haben bereits einen aktiven Auftrag ({code} — {model} am {date}). Geben Sie ihn frei, bevor Sie einen neuen beanspruchen.",
        "Español": "Ya tienes un trabajo activo ({code} — {model} el {date}). Libéralo antes de reclamar uno nuevo.",
        "Slovenščina": "Že imate aktiven posel ({code} — {model} dne {date}). Sprostite ga, preden prevzamete novega.",
    },
    "claim_button": {
        "English": "Claim",
        "Italiano": "Rivendica",
        "Deutsch": "Beanspruchen",
        "Español": "Reclamar",
        "Slovenščina": "Prevzemi",
    },
    "claimed_msg": {
        "English": "{code} claimed.",
        "Italiano": "{code} rivendicato.",
        "Deutsch": "{code} beansprucht.",
        "Español": "{code} reclamado.",
        "Slovenščina": "{code} prevzeta.",
    },
    "could_not_claim": {
        "English": "Could not claim — someone else may have taken it.",
        "Italiano": "Impossibile rivendicare — qualcun altro potrebbe averlo già preso.",
        "Deutsch": "Konnte nicht beansprucht werden — jemand anderes hat ihn möglicherweise übernommen.",
        "Español": "No se pudo reclamar — puede que alguien más lo haya tomado.",
        "Slovenščina": "Ni bilo mogoče prevzeti — morda ga je prevzel nekdo drug.",
    },
    "reason_for_declining": {
        "English": "Reason for declining",
        "Italiano": "Motivo del rifiuto",
        "Deutsch": "Grund für die Ablehnung",
        "Español": "Motivo del rechazo",
        "Slovenščina": "Razlog za zavrnitev",
    },
    "reason_declining_placeholder": {
        "English": "e.g. Too far, schedule conflict, missing equipment...",
        "Italiano": "es. Troppo lontano, conflitto di orario, attrezzatura mancante...",
        "Deutsch": "z. B. Zu weit entfernt, Terminkonflikt, fehlende Ausrüstung...",
        "Español": "p. ej. Demasiado lejos, conflicto de horario, falta de equipo...",
        "Slovenščina": "npr. Predaleč, časovni konflikt, manjkajoča oprema...",
    },
    "submit_button": {
        "English": "Submit",
        "Italiano": "Invia",
        "Deutsch": "Absenden",
        "Español": "Enviar",
        "Slovenščina": "Pošlji",
    },
    "declined_msg": {
        "English": "{code} declined — the manager has been notified.",
        "Italiano": "{code} rifiutato — il manager è stato avvisato.",
        "Deutsch": "{code} abgelehnt — der Manager wurde benachrichtigt.",
        "Español": "{code} rechazado — se notificó al gerente.",
        "Slovenščina": "{code} zavrnjena — vodja je bil obveščen.",
    },
    "please_explain_declining": {
        "English": "Please provide a short explanation before declining.",
        "Italiano": "Fornisci una breve spiegazione prima di rifiutare.",
        "Deutsch": "Bitte geben Sie vor der Ablehnung eine kurze Erklärung ab.",
        "Español": "Proporciona una breve explicación antes de rechazar.",
        "Slovenščina": "Pred zavrnitvijo navedite kratko pojasnilo.",
    },
    # Client booking form — synthetic message sent into the chat on submit
    "pending_estimate_title": {
        "English": "Check a pending estimate",
        "Italiano": "Controlla un preventivo in attesa",
        "Deutsch": "Ausstehendes Angebot prüfen",
        "Español": "Consultar un presupuesto pendiente",
        "Slovenščina": "Preveri čakajočo ponudbo",
    },
    # Client email verification (one-time code) — required only for existing tickets
    "verify_email_hint": {
        "English": "To view or respond to your tickets, first verify your email — we'll send you a 6-digit code.",
        "Italiano": "Per vedere o rispondere ai tuoi ticket, verifica prima la tua email: ti invieremo un codice di 6 cifre.",
        "Deutsch": "Um Ihre Tickets zu sehen oder zu beantworten, bestätigen Sie zuerst Ihre E-Mail — wir senden Ihnen einen 6-stelligen Code.",
        "Español": "Para ver o responder a tus tickets, verifica primero tu correo: te enviaremos un código de 6 dígitos.",
        "Slovenščina": "Za ogled ali odgovor na vaše zahtevke najprej potrdite e-pošto — poslali vam bomo 6-mestno kodo.",
    },
    "verify_send_code": {
        "English": "Send code",
        "Italiano": "Invia codice",
        "Deutsch": "Code senden",
        "Español": "Enviar código",
        "Slovenščina": "Pošlji kodo",
    },
    "verify_code_label": {
        "English": "Verification code",
        "Italiano": "Codice di verifica",
        "Deutsch": "Bestätigungscode",
        "Español": "Código de verificación",
        "Slovenščina": "Potrditvena koda",
    },
    "verify_button": {
        "English": "Verify",
        "Italiano": "Verifica",
        "Deutsch": "Bestätigen",
        "Español": "Verificar",
        "Slovenščina": "Potrdi",
    },
    "verify_code_sent": {
        "English": "We sent a code to {email}. It expires in 10 minutes.",
        "Italiano": "Abbiamo inviato un codice a {email}. Scade tra 10 minuti.",
        "Deutsch": "Wir haben einen Code an {email} gesendet. Er läuft in 10 Minuten ab.",
        "Español": "Hemos enviado un código a {email}. Caduca en 10 minutos.",
        "Slovenščina": "Kodo smo poslali na {email}. Poteče v 10 minutah.",
    },
    "verify_demo_code": {
        "English": "Demo mode — email is not configured. Your code is: {code}",
        "Italiano": "Modalità demo — email non configurata. Il tuo codice è: {code}",
        "Deutsch": "Demo-Modus — E-Mail ist nicht konfiguriert. Ihr Code lautet: {code}",
        "Español": "Modo demo — el correo no está configurado. Tu código es: {code}",
        "Slovenščina": "Demo način — e-pošta ni nastavljena. Vaša koda je: {code}",
    },
    "verify_cooldown": {
        "English": "Please wait a minute before requesting a new code.",
        "Italiano": "Attendi un minuto prima di richiedere un nuovo codice.",
        "Deutsch": "Bitte warten Sie eine Minute, bevor Sie einen neuen Code anfordern.",
        "Español": "Espera un minuto antes de solicitar un nuevo código.",
        "Slovenščina": "Pred zahtevo za novo kodo počakajte minuto.",
    },
    "verify_invalid_email": {
        "English": "Please enter a valid email address.",
        "Italiano": "Inserisci un indirizzo email valido.",
        "Deutsch": "Bitte geben Sie eine gültige E-Mail-Adresse ein.",
        "Español": "Introduce una dirección de correo válida.",
        "Slovenščina": "Vnesite veljaven e-poštni naslov.",
    },
    "verify_email_unavailable": {
        "English": "Email verification is currently unavailable. Please contact us directly.",
        "Italiano": "La verifica email non è al momento disponibile. Contattaci direttamente.",
        "Deutsch": "Die E-Mail-Bestätigung ist derzeit nicht verfügbar. Bitte kontaktieren Sie uns direkt.",
        "Español": "La verificación por correo no está disponible en este momento. Contáctanos directamente.",
        "Slovenščina": "Potrjevanje e-pošte trenutno ni na voljo. Obrnite se neposredno na nas.",
    },
    "verify_failed": {
        "English": "Invalid or expired code.",
        "Italiano": "Codice non valido o scaduto.",
        "Deutsch": "Ungültiger oder abgelaufener Code.",
        "Español": "Código no válido o caducado.",
        "Slovenščina": "Neveljavna ali potekla koda.",
    },
    "verified_as": {
        "English": "Verified as {email}",
        "Italiano": "Verificato come {email}",
        "Deutsch": "Bestätigt als {email}",
        "Español": "Verificado como {email}",
        "Slovenščina": "Potrjeno kot {email}",
    },
    "verify_change_email": {
        "English": "Use a different email",
        "Italiano": "Usa un'altra email",
        "Deutsch": "Andere E-Mail verwenden",
        "Español": "Usar otro correo",
        "Slovenščina": "Uporabi drug e-naslov",
    },
    "no_pending_estimates": {
        "English": "No estimates awaiting your response were found for this email.",
        "Italiano": "Nessun preventivo in attesa di risposta trovato per questa email.",
        "Deutsch": "Für diese E-Mail wurden keine auf Antwort wartenden Angebote gefunden.",
        "Español": "No se encontraron presupuestos pendientes de respuesta para este correo.",
        "Slovenščina": "Za to e-pošto ni bilo najdenih ponudb, ki čakajo na odgovor.",
    },
    "installation_status_title": {
        "English": "Check installation status",
        "Italiano": "Controlla lo stato dell'installazione",
        "Deutsch": "Installationsstatus prüfen",
        "Español": "Consultar el estado de la instalación",
        "Slovenščina": "Preveri stanje inštalacije",
    },
    "no_installation_updates": {
        "English": "No installation updates were found for this email.",
        "Italiano": "Nessun aggiornamento sull'installazione trovato per questa email.",
        "Deutsch": "Für diese E-Mail wurden keine Installations-Updates gefunden.",
        "Español": "No se encontraron novedades de instalación para este correo.",
        "Slovenščina": "Za to e-pošto ni bilo najdenih posodobitev o inštalaciji.",
    },
    "install_outcome_approved": {
        "English": "Completed",
        "Italiano": "Completata",
        "Deutsch": "Abgeschlossen",
        "Español": "Completada",
        "Slovenščina": "Zaključeno",
    },
    "install_outcome_rejected": {
        "English": "In progress",
        "Italiano": "In corso",
        "Deutsch": "In Bearbeitung",
        "Español": "En curso",
        "Slovenščina": "V teku",
    },
    "install_outcome_approved_note": {
        "English": "Your installation has been completed and confirmed.",
        "Italiano": "La tua installazione è stata completata e confermata.",
        "Deutsch": "Ihre Installation wurde abgeschlossen und bestätigt.",
        "Español": "Tu instalación ha sido completada y confirmada.",
        "Slovenščina": "Vaša inštalacija je bila zaključena in potrjena.",
    },
    "install_outcome_rejected_note": {
        "English": "The installer's report is being reviewed further — some work is still in progress.",
        "Italiano": "Il rapporto dell'installatore è in ulteriore revisione — alcuni lavori sono ancora in corso.",
        "Deutsch": "Der Bericht des Monteurs wird weiter geprüft — einige Arbeiten sind noch im Gange.",
        "Español": "El informe del instalador sigue en revisión — parte del trabajo aún está en curso.",
        "Slovenščina": "Poročilo monterja je še v pregledu — nekaj del je še v teku.",
    },
    "accept_estimate_button": {
        "English": "Accept estimate",
        "Italiano": "Accetta preventivo",
        "Deutsch": "Angebot annehmen",
        "Español": "Aceptar presupuesto",
        "Slovenščina": "Sprejmi ponudbo",
    },
    "preferred_install_datetime_hint": {
        "English": "Choose your preferred installation date and time.",
        "Italiano": "Scegli data e ora preferite per l'installazione.",
        "Deutsch": "Wählen Sie Ihr bevorzugtes Installationsdatum und die Uhrzeit.",
        "Español": "Elige la fecha y hora preferidas para la instalación.",
        "Slovenščina": "Izberite želeni datum in uro montaže.",
    },
    "booking_form_synthetic_message": {
        "English": "I would like to request a site survey with these details:\n- Name: {name}\n- Address: {address}\n- Model: {model}\n- Quantity: {quantity}\n- Survey date: {date}\n- Survey time: {time}\n- Email: {email}\n- Phone: {phone}",
        "Italiano": "Vorrei richiedere un sopralluogo con questi dati:\n- Nome: {name}\n- Indirizzo: {address}\n- Modello: {model}\n- Quantità: {quantity}\n- Data sopralluogo: {date}\n- Ora sopralluogo: {time}\n- Email: {email}\n- Telefono: {phone}",
        "Deutsch": "Ich möchte eine Besichtigung mit folgenden Angaben anfragen:\n- Name: {name}\n- Adresse: {address}\n- Modell: {model}\n- Menge: {quantity}\n- Besichtigungsdatum: {date}\n- Besichtigungszeit: {time}\n- E-Mail: {email}\n- Telefon: {phone}",
        "Español": "Me gustaría solicitar una visita técnica con estos datos:\n- Nombre: {name}\n- Dirección: {address}\n- Modelo: {model}\n- Cantidad: {quantity}\n- Fecha de la visita: {date}\n- Hora de la visita: {time}\n- Correo: {email}\n- Teléfono: {phone}",
        "Slovenščina": "Rad bi zahteval ogled z naslednjimi podatki:\n- Ime: {name}\n- Naslov: {address}\n- Model: {model}\n- Količina: {quantity}\n- Datum ogleda: {date}\n- Ura ogleda: {time}\n- E-pošta: {email}\n- Telefon: {phone}",
    },
}

_MONTH_NAMES: list[dict[str, str]] = [
    {
        "English": "January",
        "Italiano": "Gennaio",
        "Deutsch": "Januar",
        "Español": "Enero",
        "Slovenščina": "Januar",
    },
    {
        "English": "February",
        "Italiano": "Febbraio",
        "Deutsch": "Februar",
        "Español": "Febrero",
        "Slovenščina": "Februar",
    },
    {
        "English": "March",
        "Italiano": "Marzo",
        "Deutsch": "März",
        "Español": "Marzo",
        "Slovenščina": "Marec",
    },
    {
        "English": "April",
        "Italiano": "Aprile",
        "Deutsch": "April",
        "Español": "Abril",
        "Slovenščina": "April",
    },
    {
        "English": "May",
        "Italiano": "Maggio",
        "Deutsch": "Mai",
        "Español": "Mayo",
        "Slovenščina": "Maj",
    },
    {
        "English": "June",
        "Italiano": "Giugno",
        "Deutsch": "Juni",
        "Español": "Junio",
        "Slovenščina": "Junij",
    },
    {
        "English": "July",
        "Italiano": "Luglio",
        "Deutsch": "Juli",
        "Español": "Julio",
        "Slovenščina": "Julij",
    },
    {
        "English": "August",
        "Italiano": "Agosto",
        "Deutsch": "August",
        "Español": "Agosto",
        "Slovenščina": "Avgust",
    },
    {
        "English": "September",
        "Italiano": "Settembre",
        "Deutsch": "September",
        "Español": "Septiembre",
        "Slovenščina": "September",
    },
    {
        "English": "October",
        "Italiano": "Ottobre",
        "Deutsch": "Oktober",
        "Español": "Octubre",
        "Slovenščina": "Oktober",
    },
    {
        "English": "November",
        "Italiano": "Novembre",
        "Deutsch": "November",
        "Español": "Noviembre",
        "Slovenščina": "November",
    },
    {
        "English": "December",
        "Italiano": "Dicembre",
        "Deutsch": "Dezember",
        "Español": "Diciembre",
        "Slovenščina": "December",
    },
]


def t(key: str) -> str:
    """Returns the UI string for the current UX language."""
    lang = st.session_state.get("ux_lang_select", "English")
    return _TR.get(key, {}).get(lang) or _TR.get(key, {}).get("English", key)


def _month_name(month_1_indexed: int) -> str:
    lang = st.session_state.get("ux_lang_select", "English")
    entry = _MONTH_NAMES[month_1_indexed - 1]
    return entry.get(lang) or entry["English"]


# ── global CSS ──────────────────────────────────────────────────────────────
# All visual styling lives in assets/theme.css (loaded via utils/theme.py) —
# a pure-presentation stylesheet kept separate from this module's own logic.


def inject_global_css() -> None:
    from utils.theme import inject_custom_css

    inject_custom_css()


def render_page_header(title: str) -> None:
    """
    Renders a two-tier page title, sticky (stays visible regardless of scroll
    position): the company name as the bold primary line, and the page title
    (e.g. "Manager Portal") as a small tracked-uppercase subtitle beneath —
    the same lockup structure as the two-line wordmark on the real Marinig site.
    """
    # Deliberately a <div>, not an <h1> — Streamlit's own markdown renderer
    # post-processes h1-h6 tags (even raw ones from unsafe_allow_html) to add
    # anchor-link icons, and that pass drops the original `style` attribute,
    # which is what caused the low-contrast dark-on-dark-green text even with
    # an inline color. A div is untouched by that pass, so .header-title's
    # color rule in assets/theme.css applies reliably.
    st.markdown(
        f'<div class="page-header-sticky">'
        f'<div class="header-title">{COMPANY_CONTEXT["company_name"]}</div>'
        f'<div class="header-subtitle">{title}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── PDF helpers ───────────────────────────────────────────────────────────────

CATALOGUE_DIR = PROJECT_ROOT / "data" / "catalogue"
STATIC_CATALOGUE_DIR = PROJECT_ROOT / "app" / "static" / "catalogue"


def _ensure_static_pdf(pdf_path: Path) -> Path:
    STATIC_CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
    static_path = STATIC_CATALOGUE_DIR / pdf_path.name
    if (
        not static_path.exists()
        or static_path.stat().st_mtime < pdf_path.stat().st_mtime
        or static_path.stat().st_size != pdf_path.stat().st_size
    ):
        shutil.copy2(pdf_path, static_path)
    return static_path


@st.dialog("Catalogue PDF", width="large")
def _catalogue_pdf_dialog(pdf_url: str, pdf_name: str) -> None:
    st.markdown(
        f'<iframe src="{pdf_url}" width="100%" height="780" style="border:none;border-radius:4px;"></iframe>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<a href="{pdf_url}" target="_blank" style="font-size:0.85em;color:#14532D;">Open in full browser tab</a>',
        unsafe_allow_html=True,
    )


# ── cache eligibility ─────────────────────────────────────────────────────────

_CACHE_ELIGIBLE_TOOLS = {"search_technical_manuals", "general_door_knowledge_tool"}
_CACHE_INELIGIBLE_TOOLS = {
    "inventory_api_tool",
    "book_installation_tool",
    "check_calendar_availability_tool",
    "price_estimate_tool",
    "request_site_survey_tool",
    "calculate_travel_time_tool",
    "complete_job_tool",
    "send_alert_to_manager_tool",
    "send_estimate_to_client_tool",
    "update_installation_costs_tool",
    "request_inventory_reservation_tool",
    "submit_survey_measurements_tool",
    "respond_to_estimate_tool",
    "find_my_tickets_by_email_tool",
    "approve_survey_estimate_tool",
    "reject_survey_estimate_tool",
}


def _is_cache_eligible(tool_calls: list) -> bool:
    if not tool_calls:
        return False
    names = {tc.get("tool") for tc in tool_calls}
    return bool(names & _CACHE_ELIGIBLE_TOOLS) and not bool(names & _CACHE_INELIGIBLE_TOOLS)


# ── role labels ───────────────────────────────────────────────────────────────

ROLE_LABEL = {"client": "Client", "installer": "Installer", "manager": "Manager"}

# ── ticket card (shared between manager and installer views) ──────────────────

_TICKET_STATUS_LABEL = {
    "new": "NEW",
    "open": "OPEN",
    "in_progress": "IN PROGRESS",
    "awaiting_client": "AWAITING CLIENT",
    "blocked": "BLOCKED",
    "in_review": "IN REVIEW",
    "closed": "CLOSED",
    "cancelled": "CANCELLED",
}
_TICKET_STATUS_COLOR = {
    "new": "#2563EB",
    "open": "#6B7280",
    "in_progress": "#D97706",
    "awaiting_client": "#0891B2",
    "blocked": "#DC2626",
    "in_review": "#7C3AED",
    "closed": "#16A34A",
    "cancelled": "#57534E",
}


def _status_badge(ticket_status: str) -> str:
    # Border + colored text, transparent fill — status colors stay distinct
    # per-status (functional signal, unchanged), only the *treatment* moves
    # from a solid loud fill to the flat/outline language used everywhere else.
    label = (
        t(f"status_{ticket_status}")
        if ticket_status in _TICKET_STATUS_LABEL
        else ticket_status.upper()
    )
    color = _TICKET_STATUS_COLOR.get(ticket_status, "#6B7280")
    return (
        f'<span style="border:1px solid {color};color:{color};padding:1px 7px;'
        f"border-radius:3px;font-size:0.72rem;font-weight:700;"
        f'text-transform:uppercase;letter-spacing:0.08em;">{label}</span>'
    )


def _phase_badge(ticket_phase: str) -> str:
    label = (
        t(f"phase_{ticket_phase}")
        if ticket_phase in ("survey", "installation")
        else ticket_phase.upper()
    )
    return (
        f'<span style="border:1px solid var(--green,#1E4638);color:var(--green,#1E4638);padding:1px 7px;'
        f"border-radius:3px;font-size:0.72rem;font-weight:700;"
        f'text-transform:uppercase;letter-spacing:0.08em;">{label}</span>'
    )


def _type_badge(ticket_type: str) -> str:
    label = (
        t(f"ticket_type_{ticket_type}")
        if ticket_type in ("door", "window")
        else ticket_type.upper()
    )
    return (
        f'<span style="border:1px solid var(--amber,#A8712E);color:var(--amber,#A8712E);padding:1px 7px;'
        f"border-radius:3px;font-size:0.72rem;font-weight:700;"
        f'text-transform:uppercase;letter-spacing:0.08em;">{label}</span>'
    )


def _stock_dot(stock: int) -> str:
    if stock > 5:
        color, label = "#1E4638", str(stock)
    elif stock > 0:
        color, label = "#A8712E", str(stock)
    else:
        color, label = "#B3261E", "0"
    return (
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{color};margin-right:4px;vertical-align:middle;"></span>'
        f'<span style="font-size:0.8rem;">{label} in stock</span>'
    )


def render_ticket_card(ticket: dict, *, show_stock: bool = True) -> None:
    """
    Renders the common ticket card body — code, model, status badge, client
    info, address, notes, and assigned installer. Used by both the manager
    board and the installer job panels so the two views stay in sync.
    Manager view additionally shows stock availability (show_stock=True);
    installers don't see inventory, so their view omits it.
    """
    ts = ticket.get("ticket_status", "new")
    phase = ticket.get("ticket_phase", "installation")
    qty = ticket.get("quantity", 1)
    name = ticket.get("client_name") or ticket.get("requested_by", "—")

    # Top accent stripe in the status color — quick visual scan without reading the badge.
    # (No negative margins: stays safely inside the card regardless of the
    # exact padding Streamlit's bordered container applies.)
    status_color = _TICKET_STATUS_COLOR.get(ts, "#6B7280")
    st.markdown(
        f'<div style="height:3px;background:{status_color};border-radius:2px;margin-bottom:0.6rem;"></div>',
        unsafe_allow_html=True,
    )

    ticket_type = ticket.get("ticket_type", "window")
    st.markdown(
        f'<span class="mono-accent">{ticket_code(ticket["id"])}</span> — {qty}× **{ticket["model"]}** '
        f"{ticket['date_str']} {ticket['time_str']} &nbsp; "
        f"{_type_badge(ticket_type)} {_phase_badge(phase)} {_status_badge(ts)}",
        unsafe_allow_html=True,
    )

    client_line = f"**Client:** {name}"
    email = ticket.get("client_email", "")
    phone = ticket.get("client_phone", "")
    if email:
        client_line += f"  |  {email}"
    if phone:
        client_line += f"  |  {phone}"
    st.markdown(client_line)

    address = ticket.get("client_address", "")
    if address:
        st.caption(t("address_inline").format(address=address))

    if show_stock:
        product = get_product(ticket["model"])
        stock = product["stock"] if product else None
        if stock is not None:
            st.markdown(_stock_dot(stock), unsafe_allow_html=True)
        else:
            st.caption(t("stock_not_found"))

    if ticket.get("height_cm") and ticket.get("width_cm"):
        measurements = (
            f'<span class="mono-accent" style="color:#3F3F3F;font-size:0.8rem;">'
            f"{ticket['height_cm']:.0f}×{ticket['width_cm']:.0f}×{ticket.get('depth_cm') or 0:.0f}cm</span>"
        )
        if ticket.get("glass_type"):
            measurements += f' <span style="font-size:0.8rem;color:#3F3F3F;">· {t("glass_type_caption")} {ticket["glass_type"]}</span>'
        if ticket.get("estimated_price"):
            measurements += (
                f' <span class="mono-accent" style="font-size:0.8rem;color:#3F3F3F;">'
                f"· €{ticket['estimated_price']:.2f}</span>"
            )
        st.markdown(measurements, unsafe_allow_html=True)
    if ticket.get("notes"):
        st.caption(t("notes_inline").format(notes=ticket["notes"]))
    if ticket.get("assigned_to"):
        st.caption(
            t("installer_inline").format(
                installer=ticket["assigned_to"],
                date=(ticket.get("started_at") or "")[:16],
            )
        )

    log_entries = get_ticket_log(ticket["id"])
    if log_entries:
        with st.expander(t("ticket_log_label")):
            for entry in log_entries:
                line = (
                    f"`{entry['created_at'][:16]}` — **{entry['ticket_status'].replace('_', ' ').upper()}** "
                    f"by {entry['changed_by']}"
                )
                st.markdown(line)
                if entry.get("note"):
                    st.caption(entry["note"])


# ── stat tiles (Usage Analytics) ─────────────────────────────────────────────


def render_stat_tiles(tiles: list[dict]) -> None:
    """
    Renders a row of KPI stat tiles: value in Montserrat 700, label in
    uppercase tracked Montserrat 600 — flat, hairline-bordered, no shadow,
    matching the rest of the app's card language (assets/theme.css).
    Each tile is {"label": str, "value": str}.
    """
    cards = "".join(
        f'<div style="flex:1;min-width:120px;background:#EBEAE3;border:1px solid #C7C6BE;'
        f'border-radius:4px;padding:14px 16px;">'
        f"<div style=\"font-family:'Montserrat',sans-serif;font-size:0.7rem;color:#4E4D47;"
        f'text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-bottom:4px;">{tile["label"]}</div>'
        f"<div style=\"font-family:'Montserrat',sans-serif;font-weight:700;font-size:1.5rem;color:#161613;\">{tile['value']}</div>"
        f"</div>"
        for tile in tiles
    )
    st.markdown(
        f'<div style="display:flex;gap:12px;margin:0.25rem 0 1rem 0;flex-wrap:wrap;">{cards}</div>',
        unsafe_allow_html=True,
    )


# ── ticket status summary ────────────────────────────────────────────────────


def render_ticket_summary(tickets: list[dict]) -> None:
    """
    Compact stat-tile row of ticket counts by status — same visual language as
    render_stat_tiles. Used identically on the manager dashboard (all tickets)
    and the installer dashboard (only tickets assigned to that installer).
    """
    counts: dict = {}
    for tk in tickets:
        key = tk.get("ticket_status", "new")
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        st.caption(t("no_tickets_yet"))
        return
    tiles = [
        {
            "label": (
                t(f"status_{status}") if status in _TICKET_STATUS_LABEL else status.upper()
            ).upper(),
            "value": str(count),
        }
        for status, count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    render_stat_tiles(tiles)


# ── installer job briefing ───────────────────────────────────────────────────

_NO_INFO_MARKER = (
    "The provided documentation does not contain enough information to answer this reliably."
)


def render_installer_job_briefing(active_job: dict) -> None:
    """
    Renders inline INSIDE an active job's ticket card (called from the
    installer page's per-job loop, itself already inside a bordered
    container) — everything useful for that specific visit: an estimated
    time budget (on-site labor + best-effort travel time from the
    configured base location), whatever the client said in the original
    request, and — for installation-phase jobs only, not site surveys —
    installation guidance for that model pulled from the same
    installer-only RAG collections as search_installation_manuals_tool.
    Retrieval + LLM calls are cached per booking_id/language so they only
    run once, not on every rerun.
    """
    booking_id = active_job["id"]
    model = active_job.get("model", "")
    phase = active_job.get("ticket_phase", "installation")

    # ── time estimate — on-site hours source depends on phase: a survey
    # visit uses survey_hours_per_unit, an installation visit uses the
    # product's installation_hours. Using the wrong one would understate/
    # overstate the visit length. ─────────────────────────────────────────
    quantity = active_job.get("quantity", 1)
    if phase == "survey":
        hours_per_unit = get_costs().get("survey_hours_per_unit", 0.5)
    else:
        product = get_product(model)
        hours_per_unit = (product or {}).get("installation_hours", 1.5)
    on_site_hours = quantity * hours_per_unit
    time_parts = [f"{on_site_hours:.1f}h {t('briefing_on_site')}"]
    try:
        from tools import _compute_travel

        address = active_job.get("client_address", "")
        if address:
            travel = _compute_travel(address)
            time_parts.append(
                f"{travel['duration_hours'] * 2:.1f}h {t('briefing_round_trip_travel')}"
            )
    except Exception:
        pass  # best-effort — no origin/geocoding available, just skip this part
    st.caption(f"{t('briefing_estimated_time')}: {' + '.join(time_parts)}")

    # ── info from the client's original request, and — separately — the
    # surveyor's own note from the site visit (a different installer's
    # observation, not something the client said, so it gets its own
    # label rather than being folded into "from the client's request"). ──
    if active_job.get("notes"):
        st.caption(t("briefing_client_request"))
        st.markdown(f"> {active_job['notes']}")
    if active_job.get("survey_notes"):
        st.caption(t("briefing_surveyor_note"))
        st.markdown(f"> {active_job['survey_notes']}")

    # ── installation guidance — installation-phase jobs only. A site
    # survey visit doesn't involve installing anything, so this section
    # (and its RAG/LLM cost) is skipped entirely for those. ───────────────
    if phase == "installation":
        lang = st.session_state.get("ux_lang_select", "English")
        cache_key = _k(f"briefing_{booking_id}_{model}_{lang}")
        if cache_key not in st.session_state:
            try:
                from rag.rag_engine import get_assistant_response
                from tools import _INSTALLATION_COLLECTIONS

                # Phrased around "general procedures... like model X" rather than
                # "how do I install model X" — the installation manuals are generic
                # manufacturer procedure guides, not per-SKU documents, so a query
                # naming only the exact model tends to retrieve nothing useful.
                query = t("briefing_install_query").format(model=model)
                answer, _ = get_assistant_response(
                    query, allowed_collections=_INSTALLATION_COLLECTIONS
                )
                # If the model has no dedicated instructions but the answer still
                # goes on to give general guidance that does apply, swap the blunt
                # "not enough information" opener for an explicit, friendlier one
                # instead of leading with what sounds like a dead end.
                if answer.startswith(_NO_INFO_MARKER) and len(answer) > len(_NO_INFO_MARKER) + 20:
                    answer = (
                        t("briefing_no_specific_guidance").format(model=model)
                        + answer[len(_NO_INFO_MARKER) :]
                    )
                st.session_state[cache_key] = answer
            except Exception as exc:
                st.session_state[cache_key] = t("briefing_guidance_unavailable").format(error=exc)

        with st.expander(t("briefing_installation_guidance").format(model=model)):
            st.markdown(st.session_state[cache_key])


# ── draggable chat/dashboard column resize handle ─────────────────────────────


def render_column_resize_handle() -> None:
    """
    Injects a thin draggable handle between the chat column and dashboard
    column (manager/installer pages) so their widths can be adjusted by hand.

    st.columns() has no native resize support, so this reaches into the
    parent document from a zero-height st.components.v1.html iframe (the
    standard trick for DOM tweaks Streamlit itself doesn't expose — same
    origin, so window.parent.document is reachable) and manipulates the two
    [data-testid="stColumn"] elements' flex-basis directly via plain JS.
    Set via style.setProperty(..., 'important') rather than plain style.flex
    assignment — assets/theme.css pins these same columns' flex to fill
    their row (needed for the independent-scroll layout) with !important,
    which would otherwise silently win over a non-important inline style.

    The columns are recreated by Streamlit on every rerun, so:
    - the mousemove/mouseup listeners are attached to the parent document
      ONCE ONLY (guarded by a flag stored on that document), and always look
      up the *current* column elements fresh rather than holding onto
      references from a previous rerun that no longer exist;
    - the chosen ratio is persisted to localStorage and re-applied by a
      short-lived poll loop every time this component remounts, so the
      layout doesn't snap back to the default 3:2 split after every rerun
      (e.g. after sending a chat message).
    """
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;

            if (!doc.__chatDashboardResizeInit) {
                doc.__chatDashboardResizeInit = true;
                doc.__chatDashboardDragging = false;

                doc.addEventListener('mousemove', function(e) {
                    if (!doc.__chatDashboardDragging) return;
                    const zone = doc.querySelector('.st-key-chat_zone');
                    if (!zone) return;
                    const row = zone.closest('[data-testid="stHorizontalBlock"]');
                    if (!row) return;
                    const cols = row.querySelectorAll(':scope > [data-testid="stColumn"]');
                    if (cols.length < 2) return;
                    const rect = row.getBoundingClientRect();
                    let ratio = (e.clientX - rect.left) / rect.width;
                    ratio = Math.min(0.8, Math.max(0.2, ratio));
                    cols[0].style.setProperty('flex', ratio + ' 1 0', 'important');
                    cols[1].style.setProperty('flex', (1 - ratio) + ' 1 0', 'important');
                    localStorage.setItem('chat_dashboard_ratio', ratio.toString());
                });
                doc.addEventListener('mouseup', function() {
                    doc.__chatDashboardDragging = false;
                });
            }

            function setup() {
                const zone = doc.querySelector('.st-key-chat_zone');
                if (!zone) return false;
                const row = zone.closest('[data-testid="stHorizontalBlock"]');
                if (!row) return false;
                const cols = row.querySelectorAll(':scope > [data-testid="stColumn"]');
                if (cols.length < 2) return false;

                const stored = localStorage.getItem('chat_dashboard_ratio');
                if (stored) {
                    const ratio = parseFloat(stored);
                    cols[0].style.setProperty('flex', ratio + ' 1 0', 'important');
                    cols[1].style.setProperty('flex', (1 - ratio) + ' 1 0', 'important');
                }

                if (row.querySelector(':scope > .resize-handle')) return true;

                const handle = doc.createElement('div');
                handle.className = 'resize-handle';
                handle.style.cssText =
                    'width:6px;flex:0 0 auto;cursor:col-resize;' +
                    'background:transparent;z-index:50;position:relative;';
                handle.addEventListener('mouseenter', function() {
                    handle.style.background = 'rgba(42,87,72,0.35)';
                });
                handle.addEventListener('mouseleave', function() {
                    handle.style.background = 'transparent';
                });
                handle.addEventListener('mousedown', function(e) {
                    doc.__chatDashboardDragging = true;
                    e.preventDefault();
                });
                cols[0].after(handle);
                return true;
            }

            let attempts = 0;
            const interval = setInterval(function() {
                attempts++;
                if (setup() || attempts > 40) clearInterval(interval);
            }, 250);
        })();
        </script>
        """,
        height=0,
    )


def render_chat_autoscroll() -> None:
    """
    Jumps the chat pane to its latest message on every rerun (new user
    message, new assistant reply, HITL confirmation, etc.) — same
    reach-into-the-parent-document trick as render_column_resize_handle,
    since .st-key-chat_zone's scroll position isn't anything Streamlit
    itself exposes. Without this, a reply that's taller than the pane
    leaves the view sitting wherever it was (usually the top, from before
    the message was sent) instead of showing the message that just arrived.

    Skipped while the booking form is open: the form is often taller than
    the pane, and jumping to its bottom hides its own title and the
    name/address fields above the fold instead of showing what the client
    actually needs to see and fill in first.
    """
    if _get("show_booking_form"):
        return
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            const zone = doc.querySelector('.st-key-chat_zone');
            if (zone) {
                zone.scrollTop = zone.scrollHeight;
            }
        })();
        </script>
        """,
        height=0,
    )


# ── ticket map ────────────────────────────────────────────────────────────────


def _hex_to_rgb(hex_color: str) -> list[int]:
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]


def render_ticket_map(tickets: list[dict]) -> None:
    """
    Plots open/in-progress tickets as colored dots on a map using pydeck (a
    ScatterplotLayer with pixel-clamped radius, so dots stay small and legible
    at any zoom level — st.map's native size channel scales in meters and
    balloons to neighborhood-sized circles when zoomed in). Color = ticket
    status (same palette as the status badge); radius = phase (installation
    jobs render slightly larger than site surveys) as a second, independent
    channel. Hovering a dot shows a native tooltip with the ticket's details.
    Closed/cancelled tickets are excluded — they're no longer operationally
    relevant to plot.

    Tickets missing geocoded coordinates (client_lat/client_lon — set once at
    request time in request_site_survey_tool; tickets that predate this
    feature are simply null) are backfilled lazily here, once, via the same
    Nominatim lookup and persisted back to the DB (update_ticket_coordinates)
    so later renders don't repeat the lookup. If the full address doesn't
    resolve, a second attempt strips the house number and retries with just
    the street/city — Nominatim's free-text search is picky about exact
    tokenization, and a terse or slightly-off house number is a common
    reason the full address alone fails.

    A ticket that still has no coordinates after that (bad/incomplete
    address, or no address on file at all) is surfaced in a warning listing
    it by ticket code instead of just silently not appearing as a dot —
    the previous behavior gave the manager no way to know a ticket was
    missing from the map at all, let alone why.
    """
    active_tickets = [t for t in tickets if t.get("ticket_status") not in ("closed", "cancelled")]

    missing = [
        t
        for t in active_tickets
        if not (t.get("client_lat") and t.get("client_lon")) and t.get("client_address")
    ]
    if missing:
        from backend.database import update_ticket_coordinates
        from tools import _geocode_location

        for i, ticket in enumerate(missing):
            if i > 0:
                time.sleep(1)  # Nominatim usage policy: max ~1 request/second
            address = ticket["client_address"]
            candidates = [address]
            # Strip a trailing house number from just the first comma-separated
            # segment (the street line) — e.g. "Montenero 11, Cividale del
            # Friuli" -> "Montenero, Cividale del Friuli". Only that segment,
            # not a blind first-digit-anywhere match, so a numeric postcode or
            # province elsewhere in the string is left untouched.
            segments = [seg.strip() for seg in address.split(",")]
            if segments and re.search(r"\d", segments[0]):
                segments[0] = re.sub(r"\s*\d+[a-zA-Z]?\s*$", "", segments[0]).strip()
            stripped = ", ".join(seg for seg in segments if seg)
            if stripped and stripped != address:
                candidates.append(stripped)
            for attempt, candidate in enumerate(candidates):
                try:
                    if attempt > 0:
                        time.sleep(1)
                    lat, lon, _ = _geocode_location(candidate)
                    ticket["client_lat"], ticket["client_lon"] = lat, lon
                    update_ticket_coordinates(ticket["id"], lat, lon)
                    break
                except Exception as exc:
                    logging.getLogger("ui_shared").warning(f"MAP_BACKFILL_GEOCODE_FAILED: {exc}")

    unplaced = [tk for tk in active_tickets if not (tk.get("client_lat") and tk.get("client_lon"))]
    if unplaced:
        no_address_label = t("map_no_address")
        lines = [
            f"- {ticket_code(tk['id'])} — {tk.get('client_address') or no_address_label}"
            for tk in unplaced
        ]
        st.warning(t("map_unplaced_tickets_warning") + "\n" + "\n".join(lines))

    def _map_label(ticket: dict) -> str:
        type_label = t(f"ticket_type_{ticket.get('ticket_type', 'window')}")
        return (
            f"{ticket_code(ticket['id'])} — {ticket.get('model', '')} ({type_label})<br>"
            f"{ticket.get('client_name') or ticket.get('requested_by', '')}<br>"
            f"{_TICKET_STATUS_LABEL.get(ticket.get('ticket_status'), ticket.get('ticket_status', ''))}"
        )

    rows = [
        {
            "lat": ticket["client_lat"],
            "lon": ticket["client_lon"],
            "color": _hex_to_rgb(_TICKET_STATUS_COLOR.get(ticket.get("ticket_status"), "#6B7280")),
            "radius": 35 if ticket.get("ticket_phase") == "installation" else 22,
            "label": _map_label(ticket),
        }
        for ticket in active_tickets
        if ticket.get("client_lat") and ticket.get("client_lon")
    ]
    if not rows:
        st.info(t("map_no_locations"))
        return

    avg_lat = sum(r["lat"] for r in rows) / len(rows)
    avg_lon = sum(r["lon"] for r in rows) / len(rows)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=rows,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=6,
        radius_max_pixels=14,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=10),
        map_style="light",
        tooltip={
            "html": "<b>{label}</b>",
            "style": {
                "backgroundColor": "#161613",
                "color": "#F3F1E9",
                "fontSize": "0.8rem",
                "fontFamily": "Montserrat, sans-serif",
            },
        },
    )
    st.pydeck_chart(deck)


# ── HITL text/voice intent detection ─────────────────────────────────────────

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


# ── session defaults & helpers ────────────────────────────────────────────────

_CHAT_DEFAULTS = {
    "messages": [],
    "tool_calls_history": [],
    "total_cost": 0.0,
    "total_tokens": 0,
    "last_query_cost": 0.0,
    "total_requests": 0,
    "last_request_time": 0.0,
    "thread_id": None,
    "pending_interrupt": None,
    "audio_store": {},  # {msg_idx: mp3_bytes} for voice playback
    "autoplay_idx": -1,  # index of the message that should autoplay on next render
}


def _k(base_key: str) -> str:
    role = st.session_state.get("current_page_role", "unknown")
    return f"{role}__{base_key}"


def _get(key: str, default=None):
    return st.session_state.get(_k(key), default)


def _set(key: str, value):
    st.session_state[_k(key)] = value


# ── auth gate ─────────────────────────────────────────────────────────────────


def require_auth(allowed_roles: list[str] = None):
    inject_global_css()
    role = st.session_state.get("role")
    if not role:
        st.warning(t("please_login"))
        if st.button(t("go_login")):
            st.switch_page("pages/login.py")
        st.stop()
    if allowed_roles and role not in allowed_roles:
        st.error(t("access_denied").format(roles=", ".join(allowed_roles)))
        st.stop()
    bind_request_context()


def bind_request_context() -> None:
    """
    Copies this session's identity into the ContextVars tools read
    (username_var, verified_client_email_var). Called on every page run via
    require_auth(), so direct tool calls from dashboard buttons get the right
    identity too — not only agent turns.
    """
    username_var.set(st.session_state.get("username", "unknown"))
    verified_client_email_var.set(st.session_state.get("verified_client_email", ""))


# ── agent cache ───────────────────────────────────────────────────────────────


# Shared by every session of a role/model — safe only because the agent keeps
# no per-request state (each turn's metadata goes into its own TurnResult).
@st.cache_resource(show_spinner=False)
def get_agent(model_name: str, role: str):
    return initialize_agent(model_name=model_name, role=role)


# ── per-role session management ───────────────────────────────────────────────


def init_chat_session(role: str):
    st.session_state.current_page_role = role
    for key, default in _CHAT_DEFAULTS.items():
        if _k(key) not in st.session_state:
            v = (
                dict(default)
                if isinstance(default, dict)
                else (list(default) if isinstance(default, list) else default)
            )
            st.session_state[_k(key)] = v
    if not _get("thread_id"):
        _set("thread_id", str(uuid.uuid4()))


def reset_chat_session():
    for key, default in _CHAT_DEFAULTS.items():
        v = (
            dict(default)
            if isinstance(default, dict)
            else (list(default) if isinstance(default, list) else default)
        )
        st.session_state[_k(key)] = v
    _set("thread_id", str(uuid.uuid4()))


# ── sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar(role: str) -> str:
    """Renders the shared sidebar. Returns the active model ID."""
    model_ids = list(OPENROUTER_MODELS.keys())
    default_idx = model_ids.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_ids else 0

    with st.sidebar:
        st.markdown(
            f"**{ROLE_LABEL.get(role, role).upper()}**"
            f"  —  {st.session_state.get('username', 'Guest')}"
        )
        if st.button(t("logout"), use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.switch_page("pages/login.py")

        # ── UX language — global key so all roles share the same choice ───────
        st.divider()
        st.selectbox(
            t("language_label"),
            _LANGS,
            index=_LANGS.index(st.session_state.get("ux_lang_select", "English")),
            key="ux_lang_select",  # intentionally NOT role-namespaced
        )

        # ── model selector (manager only) ─────────────────────────────────────
        if role == "manager":
            st.divider()
            st.caption(t("model_label").upper())
            model_options = [
                f"{info['name']} ({info['provider']})" for info in OPENROUTER_MODELS.values()
            ]
            current_model_id = get_config("current_model", DEFAULT_MODEL)
            current_model_idx = (
                model_ids.index(current_model_id) if current_model_id in model_ids else default_idx
            )
            selected_display = st.selectbox(
                t("active_llm_label"),
                model_options,
                index=current_model_idx,
                label_visibility="collapsed",
            )
            new_model_id = model_ids[model_options.index(selected_display)]

            if new_model_id != current_model_id:
                set_config("current_model", new_model_id)
                reset_chat_session()
                st.rerun()

            model_info = OPENROUTER_MODELS[new_model_id]
            st.caption(f"{model_info['provider']} · {model_info['context_window']:,} tokens")
            selected_model = new_model_id
        else:
            selected_model = get_config("current_model", DEFAULT_MODEL)

        # ── voice mode ────────────────────────────────────────────────────────
        st.divider()
        voice_on = st.toggle(
            t("voice_label"), value=_get("voice_mode", False), key=_k("voice_toggle")
        )
        _set("voice_mode", voice_on)

        # ── conversation controls ─────────────────────────────────────────────
        st.divider()
        if st.button(t("clear"), use_container_width=True):
            reset_chat_session()
            st.rerun()

        # ── catalogue PDF ─────────────────────────────────────────────────────
        st.divider()
        catalogue_pdfs = sorted(CATALOGUE_DIR.glob("*.pdf"))
        if catalogue_pdfs:
            selected_pdf = st.selectbox(
                t("catalogue_select_label"),
                catalogue_pdfs,
                format_func=lambda p: p.name,
                key=_k("catalogue_pdf_select"),
                label_visibility="collapsed",
            )
            static_pdf = _ensure_static_pdf(selected_pdf)
            pdf_url = f"/app/static/catalogue/{_url_quote(static_pdf.name)}"
            if st.button(t("view_catalogue"), use_container_width=True):
                _catalogue_pdf_dialog(pdf_url, selected_pdf.name)

    return selected_model


# ── chat avatars ──────────────────────────────────────────────────────────────


def _avatar_svg(bg_hex: str, fg_hex: str, glyph: str) -> str:
    """Small inline circular avatar (data URI) — a minimal, on-brand
    alternative to Streamlit's default chat icons."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">'
        f'<circle cx="16" cy="16" r="16" fill="{bg_hex}"/>'
        f'<text x="16" y="21" font-family="Inter, sans-serif" font-size="12" font-weight="600" '
        f'fill="{fg_hex}" text-anchor="middle">{glyph}</text>'
        "</svg>"
    )
    return "data:image/svg+xml;utf8," + _url_quote(svg)


_ASSISTANT_AVATAR = _avatar_svg("#14532D", "#F0F0EE", "AI")


def _initials(name: str) -> str:
    parts = [p for p in re.split(r"[\s_.-]+", name) if p]
    if not parts:
        return "U"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def _chat_avatar(role: str) -> str:
    if role == "assistant":
        return _ASSISTANT_AVATAR
    username = st.session_state.get("username", "U")
    return _avatar_svg("#3F3F3F", "#F0F0EE", _initials(username))


# ── chat history ──────────────────────────────────────────────────────────────


def render_chat_history(agent, height: int = 560):
    """
    Renders chat messages. No fixed-height scroll container of its own —
    the chat column itself (app/ui_shared.py's `.st-key-chat_zone`-equivalent,
    the plain st.column) is the one scrollable region for the whole left
    pane (header/HITL banner/messages/input together), via CSS in
    assets/theme.css, so there's exactly one scrollbar for chat, not a nested
    scrollbar-within-a-scrollbar. `height` is accepted but unused, kept so
    existing call sites don't need updating.
    """
    audio_store = _get("audio_store", {})
    autoplay_idx = _get("autoplay_idx", -1)
    voice_mode = _get("voice_mode", False)

    with st.container(border=False):
        for msg_idx, message in enumerate(_get("messages", [])):
            with st.chat_message(message["role"], avatar=_chat_avatar(message["role"])):
                st.markdown(message["content"])

                if message["role"] == "assistant":
                    _render_answer_metadata(message)

                    # Voice audio playback
                    if voice_mode and msg_idx in audio_store:
                        should_autoplay = msg_idx == autoplay_idx
                        if should_autoplay:
                            _set("autoplay_idx", -1)
                        st.audio(audio_store[msg_idx], format="audio/mp3", autoplay=should_autoplay)
                        if st.button(t("replay"), key=_k(f"replay_{msg_idx}")):
                            _set("autoplay_idx", msg_idx)
                            st.rerun()


# ── HITL + input ──────────────────────────────────────────────────────────────


def render_hitl_and_input(agent, selected_model: str, prompt_hint: str = "Type your message..."):
    pending = _get("pending_interrupt")
    if pending:
        st.warning(f"**{t('action_required')}:** {pending.get('summary', '')}")
        col_ok, col_cancel, _ = st.columns([2, 2, 6])
        with col_ok:
            if st.button(t("confirm"), type="primary", key=_k("hitl_confirm")):
                _stream_resume(agent, "confirmed", selected_model)
        with col_cancel:
            if st.button(t("cancel"), key=_k("hitl_cancel")):
                _stream_resume(agent, "rejected", selected_model)
        st.caption(t("confirm_cancel_hint"))

    if _get("voice_mode"):
        # Audio input always available — works for normal messages AND HITL confirmations.
        audio_key = _k(f"audio_input_{_get('total_requests', 0)}")
        audio_data = st.audio_input(t("recording_hint"), key=audio_key)
        if audio_data:
            from backend.voice import transcribe_audio

            with st.spinner(t("transcribing")):
                try:
                    text = transcribe_audio(audio_data.read())
                    if text:
                        st.caption(f'"{text}"')
                        return text
                    st.warning(t("no_speech_detected"))
                except Exception as exc:
                    st.error(t("transcription_failed").format(error=exc))
        return None

    # Chat input always enabled — HITL intent is resolved in handle_user_turn.
    return st.chat_input(prompt_hint)


# ── structured booking form ──────────────────────────────────────────────────

_BOOKING_FORM_FIELD_KEYS = [
    "booking_form_client_name",
    "booking_form_client_address",
    "booking_form_model",
    "booking_form_quantity",
    "booking_form_day",
    "booking_form_month",
    "booking_form_hour",
    "booking_form_minute",
    "booking_form_client_email",
    "booking_form_client_phone",
]


def _clear_booking_form_fields() -> None:
    """
    Clears every booking-form field from session state so the form starts
    blank the next time it opens. Without this, a value the client never
    re-entered (most dangerously the email — the natural key upsert_client
    matches on) silently carries over from a previous, unrelated booking in
    the same browser session and overwrites that earlier client's row
    instead of creating a new one.
    """
    for field in _BOOKING_FORM_FIELD_KEYS:
        st.session_state.pop(_k(field), None)


def render_booking_form(agent, selected_model: str) -> None:
    """
    Compact structured booking request, shown inline in the conversation only
    after the client has expressed booking intent (see _detect_booking_intent
    in handle_user_turn, which sets the show flag and reruns) — there is no
    standalone "open form" button anywhere else on the page.

    Fields are auto-filled once, the first time the form opens, from whatever
    has already been said in the conversation (extract_booking_fields run over
    the chat transcript) — there is no separate free-text/voice box, since the
    standard chat already covers that; anything not mentioned is simply left
    for manual entry, and anything auto-filled can still be edited manually
    before sending.

    Submission feeds one consolidated message into the normal agent turn
    (handle_user_turn) — the agent still runs its own HITL confirmation
    exactly as the conversational booking path does.
    """
    show_key = _k("show_booking_form")
    if not st.session_state.get(show_key):
        return

    model_options = sorted({item["model"] for item in get_inventory()})

    autofill_key = _k("booking_form_autofilled")
    if not st.session_state.get(autofill_key):
        st.session_state[autofill_key] = True
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in _get("messages", []) if m.get("content")
        )
        if transcript.strip():
            from tools import extract_booking_fields

            try:
                extracted = extract_booking_fields(transcript)
            except Exception:
                extracted = {}

            if extracted.get("client_name"):
                st.session_state[_k("booking_form_client_name")] = str(extracted["client_name"])
            if extracted.get("client_address"):
                st.session_state[_k("booking_form_client_address")] = str(
                    extracted["client_address"]
                )
            if extracted.get("model"):
                candidate = str(extracted["model"]).upper().strip()
                if candidate in model_options:
                    st.session_state[_k("booking_form_model")] = candidate
            if extracted.get("quantity"):
                with contextlib.suppress(TypeError, ValueError):
                    st.session_state[_k("booking_form_quantity")] = max(
                        1, int(extracted["quantity"])
                    )
            if extracted.get("date_str"):
                try:
                    d = datetime.datetime.strptime(str(extracted["date_str"]), "%Y-%m-%d").date()
                    st.session_state[_k("booking_form_day")] = d.day
                    st.session_state[_k("booking_form_month")] = d.month
                except ValueError:
                    pass
            if extracted.get("time_str"):
                try:
                    tm = datetime.datetime.strptime(str(extracted["time_str"]), "%H:%M").time()
                    st.session_state[_k("booking_form_hour")] = tm.hour
                    st.session_state[_k("booking_form_minute")] = min(
                        [0, 15, 30, 45], key=lambda m: abs(m - tm.minute)
                    )
                except ValueError:
                    pass
            if extracted.get("client_email"):
                st.session_state[_k("booking_form_client_email")] = str(extracted["client_email"])
            if extracted.get("client_phone"):
                st.session_state[_k("booking_form_client_phone")] = str(extracted["client_phone"])
            st.rerun()

    today = datetime.date.today()

    with st.container(border=True):
        st.markdown(f"**{t('booking_form_title')}**")

        client_name = st.text_input(t("booking_form_name"), key=_k("booking_form_client_name"))
        client_address = st.text_input(
            t("booking_form_address"), key=_k("booking_form_client_address")
        )

        col_model, col_qty = st.columns([3, 1])
        with col_model:
            if model_options:
                model = st.selectbox(
                    t("booking_form_model"), model_options, key=_k("booking_form_model")
                )
            else:
                model = st.text_input(t("booking_form_model"), key=_k("booking_form_model"))
        with col_qty:
            quantity = st.number_input(
                t("booking_form_quantity"), min_value=1, step=1, key=_k("booking_form_quantity")
            )

        st.caption(t("booking_form_date"))
        col_day, col_month, col_hour, col_minute = st.columns(4)
        with col_day:
            day = st.selectbox(
                t("booking_form_day"), list(range(1, 32)), key=_k("booking_form_day")
            )
        with col_month:
            month = st.selectbox(
                t("booking_form_month"),
                list(range(1, 13)),
                format_func=_month_name,
                key=_k("booking_form_month"),
            )
        with col_hour:
            hour = st.selectbox(
                t("booking_form_hour"), list(range(0, 24)), key=_k("booking_form_hour")
            )
        with col_minute:
            minute = st.selectbox(
                t("booking_form_minute"),
                [0, 15, 30, 45],
                format_func=lambda m: f"{m:02d}",
                key=_k("booking_form_minute"),
            )

        if st.button(f"⏱ {t('booking_form_asap_button')}", key=_k("booking_form_asap")):
            tomorrow = today + datetime.timedelta(days=1)
            st.session_state[_k("booking_form_day")] = tomorrow.day
            st.session_state[_k("booking_form_month")] = tomorrow.month
            st.session_state[_k("booking_form_hour")] = 9
            st.session_state[_k("booking_form_minute")] = 0
            st.rerun()

        col_email, col_phone = st.columns(2)
        with col_email:
            client_email = st.text_input(
                t("booking_form_email"), key=_k("booking_form_client_email")
            )
        with col_phone:
            client_phone = st.text_input(
                t("booking_form_phone"), key=_k("booking_form_client_phone")
            )

        col_submit, col_cancel = st.columns([3, 1])
        with col_submit:
            submitted = st.button(
                t("booking_form_submit"),
                type="primary",
                key=_k("booking_form_submit"),
                use_container_width=True,
            )
        with col_cancel:
            if st.button(t("cancel"), key=_k("booking_form_close"), use_container_width=True):
                st.session_state[show_key] = False
                st.session_state[autofill_key] = False
                _clear_booking_form_fields()
                st.rerun()

        if submitted:
            required = {
                t("booking_form_name"): client_name,
                t("booking_form_address"): client_address,
                t("booking_form_email"): client_email,
                t("booking_form_phone"): client_phone,
            }
            missing = [label for label, value in required.items() if not value.strip()]

            if missing:
                st.error(f"{t('booking_form_missing')}: {', '.join(missing)}")
                return

            try:
                year = (
                    today.year if datetime.date(today.year, month, day) >= today else today.year + 1
                )
                survey_date = datetime.date(year, month, day)
            except ValueError:
                st.error(t("booking_form_invalid_date"))
                return

            date_str = survey_date.strftime("%Y-%m-%d")
            time_str = f"{hour:02d}:{minute:02d}"

            synthetic_message = t("booking_form_synthetic_message").format(
                name=client_name,
                address=client_address,
                model=model,
                quantity=quantity,
                date=date_str,
                time=time_str,
                email=client_email,
                phone=client_phone,
            )
            st.session_state[show_key] = False
            st.session_state[autofill_key] = False
            _clear_booking_form_fields()
            handle_user_turn(synthetic_message, agent, selected_model, is_synthetic=True)


_VERIFY_ERROR_KEYS = {
    "cooldown": "verify_cooldown",
    "invalid_email": "verify_invalid_email",
    "email_unavailable": "verify_email_unavailable",
}


def _render_client_email_verification(key_prefix: str) -> str:
    """
    Returns the session's verified client email — or, if there is none yet,
    renders the one-time-code form and returns "". Client sessions are
    anonymous (login.py's "Continue as Client" collects nothing), so this is
    what proves ownership before any existing ticket is shown or changed.
    The result lives in st.session_state["verified_client_email"], so
    verifying in one panel unlocks the others and the chat tools too.
    """
    from backend import client_auth

    verified = st.session_state.get("verified_client_email", "")
    if verified:
        col_info, col_change = st.columns([3, 2])
        col_info.caption(t("verified_as").format(email=verified))
        if col_change.button(t("verify_change_email"), key=_k(f"{key_prefix}_verify_change")):
            for key in ("verified_client_email", "verify_pending_email", "verify_demo_code"):
                st.session_state.pop(key, None)
            bind_request_context()
            st.rerun()
        return verified

    st.caption(t("verify_email_hint"))
    email = st.text_input(
        t("booking_form_email"),
        key=_k(f"{key_prefix}_verify_email"),
        placeholder="you@example.com",
    )
    if st.button(t("verify_send_code"), key=_k(f"{key_prefix}_verify_send")):
        result = client_auth.request_code(email)
        if result.status in ("sent", "demo"):
            st.session_state["verify_pending_email"] = client_auth.normalize_email(email)
            st.session_state["verify_demo_code"] = result.demo_code
        else:
            st.error(t(_VERIFY_ERROR_KEYS[result.status]))

    pending_email = st.session_state.get("verify_pending_email")
    if pending_email:
        demo_code = st.session_state.get("verify_demo_code")
        if demo_code:
            st.warning(t("verify_demo_code").format(code=demo_code))
        else:
            st.info(t("verify_code_sent").format(email=pending_email))
        code = st.text_input(
            t("verify_code_label"), key=_k(f"{key_prefix}_verify_code"), max_chars=6
        )
        if st.button(t("verify_button"), key=_k(f"{key_prefix}_verify_submit"), type="primary"):
            if client_auth.verify_code(pending_email, code):
                st.session_state["verified_client_email"] = pending_email
                st.session_state.pop("verify_pending_email", None)
                st.session_state.pop("verify_demo_code", None)
                bind_request_context()
                st.rerun()
            else:
                st.error(t("verify_failed"))
    return ""


def render_client_pending_estimates() -> None:
    """
    Lets a client accept/reject a priced estimate with a couple of clicks,
    instead of having to return to the chat and describe what they want in
    free text. Requires a verified email (_render_client_email_verification)
    — the same ownership rule find_my_tickets_by_email_tool and
    respond_to_estimate_tool enforce.

    Calls respond_to_estimate_tool directly (bypassing the LLM agent),
    the same direct-tool-call pattern the manager dashboard uses for
    Approve/Reject actions.
    """
    with st.expander(t("pending_estimate_title")):
        email = _render_client_email_verification("estimate")
        if not email:
            return

        from backend.database import find_tickets_by_email

        tickets = find_tickets_by_email(email, ticket_status="awaiting_client")
        if not tickets:
            st.caption(t("no_pending_estimates"))
            return

        from tools import respond_to_estimate_tool

        for ticket in tickets:
            with st.container(border=True):
                st.markdown(
                    f"**{ticket_code(ticket['id'])}** — {ticket.get('quantity', 1)}× {ticket['model']} "
                    f"— **€{ticket.get('estimated_price', 0):.2f}**"
                )
                st.caption(t("preferred_install_datetime_hint"))
                col_date, col_time = st.columns(2)
                date_val = col_date.date_input(
                    t("booking_form_date"),
                    min_value=datetime.date.today(),
                    key=_k(f"est_date_{ticket['id']}"),
                )
                time_val = col_time.time_input(
                    t("booking_form_time"), key=_k(f"est_time_{ticket['id']}")
                )

                col_accept, col_reject = st.columns(2)
                with col_accept:
                    if st.button(
                        t("accept_estimate_button"),
                        key=_k(f"est_accept_{ticket['id']}"),
                        type="primary",
                        use_container_width=True,
                    ):
                        result = respond_to_estimate_tool.invoke(
                            {
                                "booking_id": ticket["id"],
                                "response": "accept",
                                "new_date_str": date_val.strftime("%Y-%m-%d"),
                                "new_time_str": time_val.strftime("%H:%M"),
                            }
                        )
                        if "accepted —" in result:
                            st.success(result)
                            st.rerun()
                        else:
                            st.error(result)
                with col_reject:
                    if st.button(
                        t("reject_button"),
                        key=_k(f"est_reject_{ticket['id']}"),
                        use_container_width=True,
                    ):
                        result = respond_to_estimate_tool.invoke(
                            {"booking_id": ticket["id"], "response": "reject"}
                        )
                        if "has been declined" in result:
                            st.warning(result)
                            st.rerun()
                        else:
                            st.error(result)


def _install_outcome_badge(approved: bool) -> str:
    color = _TICKET_STATUS_COLOR["closed"] if approved else _TICKET_STATUS_COLOR["blocked"]
    label = t("install_outcome_approved") if approved else t("install_outcome_rejected")
    return (
        f'<span style="border:1px solid {color};color:{color};padding:1px 7px;'
        f"border-radius:3px;font-size:0.72rem;font-weight:700;"
        f'text-transform:uppercase;letter-spacing:0.08em;">{label}</span>'
    )


def render_client_installation_status() -> None:
    """
    Client-facing counterpart to render_client_pending_estimates — same
    email-lookup, bordered-card style, sitting right next to it — but for
    the OTHER end of the ticket lifecycle: the installer's completion
    report once the manager has actually resolved it (approve_job_tool /
    reject_job_tool). An approved report closes the ticket and already
    triggers a thank-you email (see approve_job_tool), but email delivery
    depends on SMTP config and the client's inbox; this makes the same
    outcome checkable directly in the portal regardless. Rejected reports
    are shown too (worded as "still being finished up", not "rejected" —
    the report was sent back to the installer for rework, not the
    installation itself being refused) since the client asked to see both
    outcomes, not just the positive one.
    """
    with st.expander(t("installation_status_title")):
        email = _render_client_email_verification("install_status")
        if not email:
            return

        from backend.database import find_tickets_by_email, get_job_completions

        tickets = find_tickets_by_email(email)
        ticket_by_id = {tk["id"]: tk for tk in tickets}
        if not ticket_by_id:
            st.caption(t("no_installation_updates"))
            return

        resolved_jobs = [
            job
            for job in get_job_completions(status="approved")
            + get_job_completions(status="rejected")
            if job.get("booking_id") in ticket_by_id
        ]
        if not resolved_jobs:
            st.caption(t("no_installation_updates"))
            return

        resolved_jobs.sort(key=lambda job: job.get("resolved_at") or "", reverse=True)
        for job in resolved_jobs:
            ticket = ticket_by_id[job["booking_id"]]
            approved = job["status"] == "approved"
            with st.container(border=True):
                st.markdown(
                    f"**{ticket_code(ticket['id'])}** — {ticket.get('quantity', 1)}× {ticket['model']} "
                    f"&nbsp; {_install_outcome_badge(approved)}",
                    unsafe_allow_html=True,
                )
                st.caption(
                    t("install_outcome_approved_note")
                    if approved
                    else t("install_outcome_rejected_note")
                )
                if job.get("structured_report"):
                    st.markdown(job["structured_report"])


def _stream_resume(agent, decision: str, selected_model: str):
    bind_request_context()
    result = TurnResult()
    with st.chat_message("assistant", avatar=_chat_avatar("assistant")):
        raw_text = st.write_stream(agent.stream_resume(decision, _get("thread_id"), result))
        _save_assistant_turn(result, raw_text, selected_model)
    st.rerun()


# ── core turn handler ─────────────────────────────────────────────────────────


def handle_user_turn(user_prompt: str, agent, selected_model: str, is_synthetic: bool = False):
    """
    is_synthetic: True for messages generated by the app itself (the booking
    form's consolidated submission, an installer's completion-notes form) —
    never organic user chat text. These skip the semantic cache lookup (no
    reason to embed a one-off transactional message full of PII into the
    cache) and the booking-intent heuristic below (a synthetic booking
    submission would otherwise re-trigger "show the booking form").
    """
    bind_request_context()

    if _get("total_requests", 0) >= 50:
        st.error(t("session_limit_reached"))
        st.stop()

    if time.time() - _get("last_request_time", 0.0) < 2:
        st.warning(t("please_wait_between_messages"))
        st.stop()

    _set("last_request_time", time.time())
    _set("total_requests", _get("total_requests", 0) + 1)

    # ── HITL text/voice intent check ─────────────────────────────────────────
    pending = _get("pending_interrupt")
    if pending:
        intent = _detect_hitl_intent(user_prompt)
        if intent:
            # Show user's confirmation/cancellation in chat history, then resume.
            msgs = _get("messages", [])
            msgs.append({"role": "user", "content": user_prompt})
            _set("messages", msgs)
            with st.chat_message("user", avatar=_chat_avatar("user")):
                st.markdown(user_prompt)
            _stream_resume(agent, intent, selected_model)
            return
        # Ambiguous message while HITL is pending — prompt without touching the graph.
        with st.chat_message("assistant", avatar=_chat_avatar("assistant")):
            st.markdown(t("hitl_pending_prompt"))
        return

    # ── Booking-intent heuristic (client role, organic messages only) ────────
    if (
        not is_synthetic
        and st.session_state.get("role") == "client"
        and _detect_booking_intent(user_prompt)
    ):
        msgs = _get("messages", [])
        msgs.append({"role": "user", "content": user_prompt})
        msgs.append(
            {
                "role": "assistant",
                "content": t("booking_intent_redirect"),
                "source_info": None,
                "tools": [],
                "run_id": None,
                "is_general_llm": False,
                "is_cache_hit": False,
            }
        )
        _set("messages", msgs)
        st.session_state[_k("show_booking_form")] = True
        st.rerun()
        return

    # ── Normal turn ───────────────────────────────────────────────────────────
    messages = _get("messages", [])
    messages.append({"role": "user", "content": user_prompt})
    _set("messages", messages)

    with st.chat_message("user", avatar=_chat_avatar("user")):
        st.markdown(user_prompt)

    # Cache check (primary path) — skipped for synthetic/transactional messages.
    cached_answer = None if is_synthetic else _cache_lookup(user_prompt)
    if cached_answer:
        with st.chat_message("assistant", avatar=_chat_avatar("assistant")):
            st.markdown(cached_answer)
            st.success(t("cache_hit"))
        msgs = _get("messages", [])
        msgs.append(
            {
                "role": "assistant",
                "content": cached_answer,
                "source_info": None,
                "tools": [],
                "run_id": None,
                "is_general_llm": False,
                "is_cache_hit": True,
            }
        )
        _set("messages", msgs)
        st.rerun()
        return

    result = TurnResult()
    with st.chat_message("assistant", avatar=_chat_avatar("assistant")):
        raw_text = st.write_stream(agent.stream_ask(user_prompt, _get("thread_id"), result))
        _save_assistant_turn(result, raw_text, selected_model, user_question=user_prompt)

    st.rerun()


# ── answer metadata ───────────────────────────────────────────────────────────


def _catalogue_page_link(source_info: str):
    if not source_info:
        return None
    for chunk in source_info.split(" ; "):
        if "catalogue" not in chunk.lower():
            continue
        fname = re.search(r"(\S+\.pdf)", chunk)
        page = re.search(r"pages?\s+(\d+)", chunk)
        if fname and page:
            return fname.group(1), int(page.group(1))
    return None


def _render_answer_metadata(message: dict):
    if message.get("is_cache_hit"):
        st.success(t("cache_hit"))
    if message.get("source_info") and not message.get("is_cache_hit"):
        # The raw chunk/source dump (filenames, page ranges, collection
        # names) used to be shown verbatim here — removed for a cleaner
        # look; the answer itself now ends with the relevant webpage's
        # full URL when applicable (see generate_answer's prompt), and the
        # catalogue-PDF deep link below still works for PDF-sourced answers.
        cat = _catalogue_page_link(message["source_info"])
        if cat:
            filename, page_num = cat
            pdf_url = f"/app/static/catalogue/{_url_quote(filename)}#page={page_num}"
            st.markdown(
                f'<a href="{pdf_url}" target="_blank" style="font-size:0.8rem;color:#14532D;">'
                f"{t('open_catalogue')} {page_num}</a>",
                unsafe_allow_html=True,
            )
    if message.get("is_general_llm"):
        st.info(t("general_llm"))


# ── save assistant turn ───────────────────────────────────────────────────────


def _save_assistant_turn(
    result: TurnResult, raw_text: str, selected_model: str, user_question: str = ""
):
    captured_tools = result.tool_calls
    usage = result.usage
    run_id = result.run_id
    _set("pending_interrupt", result.interrupt)

    cost = calculate_cost(
        selected_model, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    )
    _set("total_cost", _get("total_cost", 0.0) + cost)
    _set("last_query_cost", cost)
    _set("total_tokens", _get("total_tokens", 0) + usage.get("total_tokens", 0))

    with contextlib.suppress(Exception):
        log_usage(
            role=st.session_state.get("role", "unknown"),
            model=selected_model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cost_usd=cost,
        )

    is_general_llm = any(tc.get("tool") == "general_door_knowledge_tool" for tc in captured_tools)
    is_cache_hit = any("CACHE_HIT" in str(tc.get("output", "")) for tc in captured_tools)

    raw_text = raw_text.replace("GENERAL_LLM_SPLIT", "").replace("CACHE_HIT", "").strip()

    if "SOURCE_DATA_SPLIT" in raw_text:
        main_answer, source_info = raw_text.split("SOURCE_DATA_SPLIT", 1)
        source_info = source_info.strip()
    else:
        main_answer, source_info = raw_text, None

    if not source_info:
        for tc in captured_tools:
            if tc.get("tool") == "search_technical_manuals":
                tool_out = str(tc.get("output", ""))
                if "SOURCE_DATA_SPLIT" in tool_out:
                    source_info = (
                        tool_out.split("SOURCE_DATA_SPLIT", 1)[1].replace("CACHE_HIT", "").strip()
                    )
                break

    main_answer = main_answer.strip()
    pending_interrupt = result.interrupt
    if not main_answer and pending_interrupt and pending_interrupt.get("summary"):
        # The graph paused for confirmation without saying anything first — use the
        # interrupt's summary as the bubble content so it's part of the permanent
        # chat history, not just the transient warning banner shown while pending.
        main_answer = pending_interrupt["summary"]

    msg = {
        "role": "assistant",
        "content": main_answer,
        "source_info": source_info,
        "tools": captured_tools,
        "run_id": run_id,
        "is_general_llm": is_general_llm,
        "is_cache_hit": is_cache_hit,
    }
    _render_answer_metadata(msg)

    # Generate TTS and store for post-rerun playback.
    # Do NOT call st.audio() here — st.rerun() would kill it immediately.
    if _get("voice_mode") and main_answer.strip():
        from backend.voice import text_to_speech

        try:
            audio_bytes = text_to_speech(main_answer)
            if audio_bytes:
                current_msgs = _get("messages", [])
                new_msg_idx = len(current_msgs)  # index this msg will occupy
                audio_store = _get("audio_store", {})
                audio_store[new_msg_idx] = audio_bytes
                _set("audio_store", audio_store)
                _set("autoplay_idx", new_msg_idx)
        except Exception:
            st.caption(t("audio_error"))

    if user_question and not is_cache_hit and _is_cache_eligible(captured_tools):
        with contextlib.suppress(Exception):
            qa_submit(
                question=user_question,
                answer=main_answer.strip(),
                source_role=st.session_state.get("role", "unknown"),
            )

    if captured_tools:
        history = _get("tool_calls_history", [])
        history.extend(captured_tools)
        _set("tool_calls_history", history)

    messages = _get("messages", [])
    messages.append(msg)
    _set("messages", messages)
