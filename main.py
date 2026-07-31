"""
WhatsApp AI Ordering System — FastAPI Backend
=============================================

A demo backend for a multi-restaurant WhatsApp ordering agent in Karachi.

Features:
  - POST /webhook  -> receives Twilio WhatsApp messages (text + voice notes)
  - GET  /orders   -> lists confirmed orders (held in memory)
  - GET  /health   -> health check
  - Per-user conversation memory (in-memory dict, no database)
  - OpenAI GPT-4o for the ordering conversation
  - Voice notes: Deepgram (speech-to-text) -> GPT -> OpenAI TTS (text-to-speech)

Run with:
    uvicorn main:app --reload --port 8000

All secrets are read from a .env file (see .env.example).
"""

import os
import re
import json
import uuid
import base64
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
from twilio.rest import Client as TwilioClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("whatsapp-ordering")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")  # e.g. "whatsapp:+14xxxxxxx6"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Public base URL of THIS server, used so Twilio can fetch generated audio files.
# e.g. "https://abc123.ngrok.io" — set this in .env when testing voice replies.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# OpenAI TTS voice — "nova" is a clear, neutral stock voice.
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")

# Languages we actually support for voice transcription. Deepgram's
# detect_language can mis-tag short/noisy audio (e.g. as Chinese/Japanese);
# anything outside this set is treated as a misdetection and handled as English.
ALLOWED_TRANSCRIPTION_LANGUAGES = {"en", "ur", "hi"}

# Directory where generated reply audio is written so it can be served back.
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Clients (created lazily-safe: only if keys are present)
# ---------------------------------------------------------------------------

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

twilio_client = (
    TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
    else None
)

# ---------------------------------------------------------------------------
# System prompt for the ordering agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """LANGUAGE RULE (HIGHEST PRIORITY — APPLIES BEFORE EVERYTHING ELSE): \
You MUST detect the language of ONLY the user's LATEST message and reply in that SAME language. \
Detect the language fresh from the LATEST message ALONE, independently — never assume it from earlier messages in the conversation. \
There are EXACTLY THREE allowed reply languages and NO others: \
- If the latest message is mostly English words → reply ONLY in English. \
- If the latest message is Roman Urdu (Urdu words written in Latin/English script, e.g. 'khana', 'mujhe', 'karo', 'chahiye', 'kitne') → reply ONLY in Roman Urdu. \
- If the latest message contains Urdu script (Arabic letters, e.g. اردو) → reply ONLY in Urdu script. \
ABSOLUTELY FORBIDDEN: You must NEVER, under ANY circumstance, reply in Chinese, Japanese, Korean, Arabic, Hindi (Devanagari), Russian, or ANY language other than the three allowed above. \
This is a hard constraint with no exceptions. \
If you are EVER uncertain which language the user used, DEFAULT TO ENGLISH. \
NEVER mix two languages in a single reply. Do NOT default to Roman Urdu — only use Roman Urdu when the latest message is actually in Roman Urdu. \
This language rule overrides all other instructions. \

You are an AI ordering agent for a multi-restaurant platform in Karachi. \
Guide the customer to order food. First ask what they want to eat, then ask their area/location, \
then show only restaurants that have that item AND deliver to their area. \
Mock restaurants: \
(1) Burger Bites — area: Gulshan, Clifton — menu: Zinger Burger (Rs. 550), Smash Burger (Rs. 750), Fries (Rs. 250). \
(2) Pizza Palace — area: DHA, Clifton — menu: Pepperoni Pizza (Rs. 1400), BBQ Pizza (Rs. 1500). \
(3) Desi Dhaba — area: Gulshan, Nazimabad — menu: Nihari (Rs. 650), Biryani (Rs. 450), Karahi (Rs. 1800). \
(4) Wrap & Roll — area: DHA, PECHS — menu: Chicken Wrap (Rs. 600), Beef Wrap (Rs. 700). \
Delivery fee is Rs. 100 on every order. \
ALWAYS ask how many (quantity) the customer wants before confirming, and ALWAYS include the quantity in the order JSON. \
Quote prices and the total bill in Rupees when confirming. \
BEFORE confirming a new order you MUST ask the customer for their NAME and their PHONE NUMBER, and wait for their answer. \
Put the name and the phone number EXACTLY as the customer typed them into the order JSON. \
NEVER invent, guess, autofill or make up a name or a phone number — if the customer has not told you their phone number yet, \
ask for it; do not put a number of your own choosing in the JSON. If you truly have no phone number from them, \
use the literal placeholder "TWILIO_NUMBER" and nothing else. \
When customer confirms order, return a special JSON in your message like: \
ORDER_CONFIRMED:{"customer_name":"...", "item":"...", "quantity":1, "restaurant":"...", "area":"...","phone":"03001234567"}

MODIFYING AN EXISTING ORDER (IMPORTANT): If the customer already has an order and wants to CHANGE it \
— a different item, a different quantity, or a different restaurant — you MUST NOT emit ORDER_CONFIRMED. \
Emit ORDER_UPDATED instead, with the COMPLETE final state of the order after the change (not just the changed field): \
ORDER_UPDATED:{"customer_name":"...", "item":"...", "quantity":1, "restaurant":"...", "area":"...","phone":"TWILIO_NUMBER"} \
Use ORDER_CONFIRMED only for a brand-new order. Use ORDER_UPDATED whenever you are changing an order that already exists. \
When modifying, CARRY OVER every detail the customer did NOT ask to change (quantity, restaurant, area, name) from the existing order — \
do NOT ask them again for details they already gave. This applies even when the item AND restaurant both change: \
keep the previous quantity unless the customer states a new one. NEVER re-ask for the quantity during a modification. \
Apply the change immediately and state the new bill.

CANCELLING AN ORDER (IMPORTANT): If the customer wants to CANCEL their order, you must ask for confirmation FIRST. \
Reply by asking exactly this, in the customer's language: "Are you sure you want to cancel your order?" \
Do NOT emit any marker at this stage — just ask the question and wait. \
ONLY after the customer clearly confirms (e.g. "yes", "haan", "confirm", "cancel it") do you emit: \
ORDER_CANCELLED:{"phone":"TWILIO_NUMBER", "item":"the exact item of the ONE order being cancelled"} \
If the customer says no / changes their mind, do NOT emit the marker — keep the order as it is and reassure them. \
Never emit ORDER_CANCELLED in the same message as the confirmation question. \
CANCEL ONLY ONE ORDER AT A TIME. If the customer has SEVERAL orders, first list them and ask WHICH ONE \
they want to cancel, then confirm, then emit the marker with that order's "item" filled in. \
The "item" field is what tells the system which single order to delete — always include it, and never \
list more than one item in it. Only if the customer explicitly says they want to cancel ALL of their \
orders may you omit "item"."""

# ---------------------------------------------------------------------------
# In-memory state (resets on restart — this is a demo, no database)
# ---------------------------------------------------------------------------

# Maps a WhatsApp sender id -> list of OpenAI chat messages (the conversation).
conversations: dict[str, list[dict]] = {}

# Confirmed orders, newest last. Exposed via GET /orders.
orders: list[dict] = []

# Cap conversation history length to keep token usage sane in a long chat.
MAX_HISTORY_MESSAGES = 30

# ---------------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="WhatsApp AI Ordering Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # demo: allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_history(sender: str) -> list[dict]:
    """Return (creating if needed) the conversation history for a sender."""
    if sender not in conversations:
        conversations[sender] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return conversations[sender]


def trim_history(history: list[dict]) -> None:
    """Keep the system prompt + the most recent messages, in place."""
    if len(history) <= MAX_HISTORY_MESSAGES:
        return
    system = history[0]
    recent = history[-(MAX_HISTORY_MESSAGES - 1):]
    history.clear()
    history.append(system)
    history.extend(recent)


def run_gpt(sender: str, user_text: str) -> str:
    """Append the user's message, call GPT-4o, store and return the reply."""
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    history = get_history(sender)
    history.append({"role": "user", "content": user_text})
    trim_history(history)

    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=history,
        temperature=0.6,
    )
    reply = completion.choices[0].message.content or ""
    history.append({"role": "assistant", "content": reply})
    return reply


def extract_order(reply: str, sender: str) -> Optional[dict]:
    """
    Look for an ORDER_CONFIRMED:{...} marker in the GPT reply.

    Returns the parsed order dict (enriched with id/phone/timestamp) if found,
    otherwise None.

    ORDER_UPDATED is also accepted here as a safety net: handle_confirmed_order
    routes real updates away before reaching this point, so a marker that gets
    this far is one we found no existing order for — recording it as a new order
    beats dropping the customer's request.
    """
    match = re.search(r"ORDER_(?:CONFIRMED|UPDATED):\s*(\{.*\})", reply, re.DOTALL)
    if not match:
        return None

    raw = match.group(1)
    try:
        order = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Found ORDER_CONFIRMED marker but JSON did not parse: %s", raw)
        return None

    # The sender id from WhatsApp/the simulator is the authoritative identity —
    # never whatever the model wrote into the JSON. Keep it separately so looking
    # up "this customer's order" can't be thrown off by a hallucinated or
    # customer-typed phone number.
    order["sender_id"] = sender
    phone = sender.replace("whatsapp:", "")
    if not is_plausible_phone(order.get("phone")):
        # Missing, still the placeholder, or not phone-shaped (the model
        # occasionally invents one) — fall back to the real sender number.
        if order.get("phone") and order.get("phone") != "TWILIO_NUMBER":
            logger.warning(
                "Discarding implausible phone %r from model output; using sender %r.",
                order.get("phone"),
                phone,
            )
        order["phone"] = phone

    order["id"] = str(uuid.uuid4())
    order["created_at"] = datetime.now(timezone.utc).isoformat()
    order["status"] = "confirmed"
    price_order(order)
    return order


MENU_PRICES = {
    "zinger burger": 550,
    "smash burger": 750,
    "fries": 250,
    "pepperoni pizza": 1400,
    "bbq pizza": 1500,
    "nihari": 650,
    "biryani": 450,
    "karahi": 1800,
    "chicken wrap": 600,
    "beef wrap": 700,
}

DELIVERY_FEE = 100
CURRENCY = "Rs."


def is_plausible_phone(phone) -> bool:
    """
    True if this looks like a phone number a customer actually gave us.

    Rejects the TWILIO_NUMBER placeholder, empty/ellipsis values, and anything
    without enough digits to be a real number — the model sometimes fills the
    field in with an invented number rather than asking.
    """
    if not phone or not isinstance(phone, str):
        return False
    value = phone.strip()
    if value in ("", "...", "TWILIO_NUMBER"):
        return False
    return len(re.sub(r"\D", "", value)) >= 10


def normalize_phone(phone: str) -> str:
    """
    Reduce a customer identifier to a stable comparison key.

    Normally this is a phone number, so strip formatting down to digits. But the
    browser simulator sends a non-numeric sender id (e.g. "test_user"), which
    would reduce to an empty string and make every order look phone-less — so
    identifiers with no digits fall back to their lowercased text form.
    """
    raw = str(phone or "").replace("whatsapp:", "").strip()
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return raw.lower()

    # Pakistani numbers arrive both locally ("03001234567") and internationally
    # ("+923001234567"); reduce both to the same key so a customer who typed the
    # local form still matches their WhatsApp sender number.
    if digits.startswith("92"):
        digits = digits[2:]
    digits = digits.lstrip("0")
    return digits


def order_quantity(order: dict) -> int:
    """Quantity on an order, defaulting to 1 when the agent omitted it."""
    try:
        qty = int(order.get("quantity") or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, qty)


def price_order(order: dict) -> dict:
    """
    Work out unit price, subtotal, delivery fee and total for an order, and
    write them onto the order dict so /orders and the dashboard can show a bill.

    Unknown items price at 0 rather than failing — this is a demo menu, and a
    missing price shouldn't stop an order from being recorded.
    """
    quantity = order_quantity(order)
    items = normalize_items(order)

    unit_price = sum(MENU_PRICES.get(item, 0) for item in items)
    subtotal = unit_price * quantity
    total = subtotal + DELIVERY_FEE if subtotal else 0

    order["quantity"] = quantity
    order["unit_price"] = unit_price
    order["subtotal"] = subtotal
    order["delivery_fee"] = DELIVERY_FEE if subtotal else 0
    order["total"] = total
    order["currency"] = CURRENCY

    if unit_price == 0 and items:
        logger.warning("No menu price found for item(s) %s — bill shows 0.", items)
    return order


def normalize_items(order: dict) -> tuple[str, ...]:
    """
    Build a comparable, order-insensitive view of the item(s) on an order.

    The agent normally emits a single "item" string, but tolerate an "items"
    list (of strings or {"name"/"item": ...} dicts) and comma/"and"-separated
    strings so duplicates are still detected.
    """
    raw = order.get("items")
    if raw is None:
        raw = order.get("item")

    if raw is None:
        values: list[str] = []
    elif isinstance(raw, str):
        values = re.split(r"\s*(?:,|\band\b|\+|&)\s*", raw)
    elif isinstance(raw, (list, tuple)):
        values = []
        for entry in raw:
            if isinstance(entry, dict):
                values.append(str(entry.get("name") or entry.get("item") or ""))
            else:
                values.append(str(entry))
    else:
        values = [str(raw)]

    cleaned = [re.sub(r"\s+", " ", v).strip().lower() for v in values]
    return tuple(sorted(v for v in cleaned if v))


def customer_key(order: dict) -> str:
    """
    Stable identity for the customer an order belongs to.

    Prefers the sender id captured from WhatsApp/the simulator, falling back to
    the phone field for orders recorded before sender_id existed.
    """
    return normalize_phone(order.get("sender_id") or order.get("phone"))


def find_duplicate_order(order: dict) -> Optional[dict]:
    """
    Return an existing active order from the same customer with the same items,
    or None if this order is new.
    """
    key = customer_key(order)
    items = normalize_items(order)
    if not key or not items:
        return None

    for existing in orders:
        if existing.get("status") in ("cancelled", "canceled"):
            continue
        if customer_key(existing) != key:
            continue
        if normalize_items(existing) == items:
            return existing
    return None


def describe_order(order: dict) -> str:
    """
    Short human-readable summary of an order, including the bill. Used in the
    duplicate and update notices so the agent can quote exact figures.
    """
    parts = [", ".join(normalize_items(order)) or "your order"]
    if order.get("quantity"):
        parts.insert(0, f"{order['quantity']}x")
    if order.get("restaurant"):
        parts.append(f"from {order['restaurant']}")
    if order.get("area"):
        parts.append(f"to {order['area']}")

    if order.get("total"):
        currency = order.get("currency", CURRENCY)
        parts.append(
            f"— bill: {currency} {order.get('subtotal', 0)} "
            f"+ {currency} {order.get('delivery_fee', 0)} delivery "
            f"= {currency} {order['total']} total"
        )
    return " ".join(parts)


# Fields the agent may change on an existing order. "id" and "created_at" are
# deliberately excluded so the order keeps its identity and original timestamp.
UPDATABLE_ORDER_FIELDS = ("item", "items", "quantity", "restaurant", "area", "customer_name")


def extract_order_update(reply: str, sender: str) -> Optional[dict]:
    """
    Look for an ORDER_UPDATED:{...} marker in the GPT reply.

    Returns the parsed patch dict (with the phone placeholder resolved) if found,
    otherwise None. Unlike extract_order this does NOT mint a new id/created_at —
    the existing order keeps its own.
    """
    match = re.search(r"ORDER_UPDATED:\s*(\{.*\})", reply, re.DOTALL)
    if not match:
        return None

    raw = match.group(1)
    try:
        patch = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Found ORDER_UPDATED marker but JSON did not parse: %s", raw)
        return None

    phone = sender.replace("whatsapp:", "")
    if not patch.get("phone") or patch.get("phone") == "TWILIO_NUMBER":
        patch["phone"] = phone
    return patch


def find_order_by_phone(sender: str) -> Optional[dict]:
    """
    Return this customer's most recent active order, or None.

    Matched on the sender id (see customer_key) rather than any phone number the
    model wrote, and searched newest-first so a modification always lands on the
    customer's latest order.
    """
    target = normalize_phone(sender)
    if not target:
        return None

    for existing in reversed(orders):
        if existing.get("status") in ("cancelled", "canceled"):
            continue
        if customer_key(existing) == target:
            return existing
    return None


def apply_order_update(existing: dict, patch: dict) -> dict:
    """
    Update an existing order in place with the changed fields.

    Mutates (and returns) the dict already held in `orders`, so the order keeps
    its original id and created_at. Only known, non-empty fields are copied, and
    "item"/"items" are kept mutually exclusive so a changed item can't leave a
    stale value behind on the other key.
    """
    changed: dict[str, tuple] = {}

    for field in UPDATABLE_ORDER_FIELDS:
        if field not in patch:
            continue
        new_value = patch[field]
        if new_value in (None, "", "...", []):
            continue
        if existing.get(field) == new_value:
            continue
        changed[field] = (existing.get(field), new_value)
        existing[field] = new_value

    # If the agent switched which key it uses for the item(s), drop the other one
    # so the order doesn't end up describing two different meals.
    if "items" in changed and "item" in existing:
        existing.pop("item", None)
    elif "item" in changed and "items" in existing:
        existing.pop("items", None)

    if changed:
        existing["status"] = "updated"
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        # Item or quantity may have changed — recompute the bill.
        price_order(existing)
    return changed


# Injected as a system message so the agent itself confirms the update in the
# customer's own language.
ORDER_UPDATED_INSTRUCTION = """SYSTEM NOTICE (not from the customer): \
The customer's EXISTING order was UPDATED in place — no new order was created. \
Order id {id} (unchanged), originally placed at {created_at}. \
The order now is: {summary}. \
Confirm to the customer that their existing order was UPDATED (not newly created), and state the new details. \
Do NOT say a new order was placed. Do NOT output any ORDER_CONFIRMED or ORDER_UPDATED marker in this reply. \
Follow the language rule: reply in the language of the customer's latest message."""

ORDER_UPDATED_FALLBACK = (
    "Your existing order has been updated — it's now {summary}. "
    "No new order was created. Anything else you'd like to change?"
)


def build_order_updated_reply(sender: str, updated: dict) -> str:
    """
    Ask the agent to confirm that the existing order was updated, not created.

    Mirrors build_duplicate_order_reply: the marker-bearing reply is dropped from
    history and replaced by this one, with a canned fallback if GPT is
    unavailable or the call fails.
    """
    history = get_history(sender)
    if len(history) > 1 and history[-1].get("role") == "assistant":
        history.pop()

    notice = ORDER_UPDATED_INSTRUCTION.format(
        id=updated.get("id", "unknown"),
        created_at=updated.get("created_at", "unknown"),
        summary=describe_order(updated),
    )
    fallback = ORDER_UPDATED_FALLBACK.format(summary=describe_order(updated))

    reply = ""
    if openai_client is not None:
        history.append({"role": "system", "content": notice})
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=history,
                temperature=0.6,
            )
            reply = completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — fall back to a canned message
            logger.exception("Order-updated reply generation failed: %s", exc)
        finally:
            history.pop()

    reply = clean_reply_for_user(reply) if reply.strip() else fallback
    history.append({"role": "assistant", "content": reply})
    trim_history(history)
    return reply


# Injected as a system message so the agent itself tells the customer (in the
# customer's own language) that the order already exists.
DUPLICATE_ORDER_INSTRUCTION = """SYSTEM NOTICE (not from the customer): \
An identical order from this same phone number already exists, so NO new order was created. \
Existing order — id: {id}, placed at: {created_at}, details: {summary}. \
Tell the customer their order already exists (mention what it is), then ask whether they want to \
MODIFY it, CANCEL it, or whether placing it again was a MISTAKE. \
Do NOT output an ORDER_CONFIRMED marker in this reply. \
Follow the language rule: reply in the language of the customer's latest message."""

DUPLICATE_ORDER_FALLBACK = (
    "You already have an order with us for {summary}, so I haven't placed a new one. "
    "Would you like to modify it, cancel it, or was placing it again a mistake?"
)


def build_duplicate_order_reply(sender: str, duplicate: dict) -> str:
    """
    Ask the agent to inform the customer that their order already exists.

    The confirmation reply is dropped from the history (the customer never sees
    it) and replaced by this notice reply, so the conversation stays consistent.
    Falls back to a canned message if GPT is unavailable or the call fails.
    """
    history = get_history(sender)
    if len(history) > 1 and history[-1].get("role") == "assistant":
        history.pop()

    notice = DUPLICATE_ORDER_INSTRUCTION.format(
        id=duplicate.get("id", "unknown"),
        created_at=duplicate.get("created_at", "unknown"),
        summary=describe_order(duplicate),
    )
    fallback = DUPLICATE_ORDER_FALLBACK.format(summary=describe_order(duplicate))

    reply = ""
    if openai_client is not None:
        history.append({"role": "system", "content": notice})
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=history,
                temperature=0.6,
            )
            reply = completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — fall back to a canned message
            logger.exception("Duplicate-order reply generation failed: %s", exc)
        finally:
            # Drop the transient notice so it doesn't linger in the history.
            history.pop()

    reply = clean_reply_for_user(reply) if reply.strip() else fallback
    history.append({"role": "assistant", "content": reply})
    trim_history(history)
    return reply


def handle_order_update(
    reply: str, sender: str, channel: str
) -> tuple[Optional[str], Optional[dict]]:
    """
    If the reply contains an ORDER_UPDATED marker, update the customer's existing
    order in place (found by phone number) instead of creating a new one.

    Returns (reply_to_send, updated_order). Both are None when there is no update
    marker, so the caller falls through to the untouched create path. If no
    existing order can be found, the marker is treated as a new order instead so
    the customer's request isn't silently dropped.
    """
    patch = extract_order_update(reply, sender)
    if not patch:
        return None, None

    existing = find_order_by_phone(sender)
    if existing is None:
        # Nothing to modify — fall back to the normal create path by rewriting
        # the marker, so the order still gets recorded.
        logger.info(
            "ORDER_UPDATED (%s) from %s but no existing order found — treating as new order.",
            channel,
            sender,
        )
        return None, None

    changed = apply_order_update(existing, patch)
    if not changed:
        logger.info(
            "ORDER_UPDATED (%s) from %s matched order %s but nothing changed.",
            channel,
            sender,
            existing.get("id"),
        )
    else:
        logger.info(
            "Order %s updated in place (%s) from %s: %s",
            existing.get("id"),
            channel,
            sender,
            {k: f"{old!r} -> {new!r}" for k, (old, new) in changed.items()},
        )
    return build_order_updated_reply(sender, existing), existing


def orders_for_phone(phone: str) -> list[dict]:
    """Return this customer's orders, oldest first, without removing anything."""
    target = normalize_phone(phone)
    if not target:
        return []
    return [
        order
        for order in orders
        if customer_key(order) == target or normalize_phone(order.get("phone")) == target
    ]


def remove_orders(to_remove: list[dict]) -> list[dict]:
    """
    Delete the given orders from the `orders` list in place, by identity.

    Compared with `is` rather than `==` so two orders that happen to hold equal
    values can't be removed together — only the exact dicts passed in go.
    """
    if not to_remove:
        return []
    remaining = [o for o in orders if not any(o is target for target in to_remove)]
    orders.clear()
    orders.extend(remaining)
    return to_remove


def select_orders_to_cancel(
    candidates: list[dict], order_id: Optional[str] = None, item: Optional[str] = None
) -> list[dict]:
    """
    Narrow a customer's orders down to the one(s) they actually named.

    Matching is tried most-specific first: an exact order id, then the item name
    (exact match, then a looser substring match so "burger" finds "Zinger
    Burger"). Returns [] when nothing matches, so the caller can ask instead of
    guessing — cancelling the wrong order is worse than asking again.
    """
    if order_id:
        wanted = str(order_id).strip().lower()
        matches = [o for o in candidates if str(o.get("id", "")).lower() == wanted]
        if matches:
            return matches

    if item:
        wanted = re.sub(r"\s+", " ", str(item)).strip().lower()
        if wanted:
            exact = [o for o in candidates if wanted in normalize_items(o)]
            if exact:
                return exact
            partial = [
                o
                for o in candidates
                if any(wanted in existing or existing in wanted for existing in normalize_items(o))
            ]
            if partial:
                return partial

    return []


def delete_orders_for_phone(
    phone: str, order_id: Optional[str] = None, item: Optional[str] = None
) -> list[dict]:
    """
    Remove this customer's orders and return the ones removed.

    With no order_id/item, every order for the phone is removed (the original
    behaviour, used by DELETE /orders/{phone}). Passing either narrows it to just
    the matching order, so cancelling one order out of several leaves the rest
    alone. Matching accepts a phone number or a full sender id, in any format.
    """
    candidates = orders_for_phone(phone)
    if not candidates:
        return []

    if order_id or item:
        selected = select_orders_to_cancel(candidates, order_id=order_id, item=item)
        if not selected:
            return []
        return remove_orders(selected)

    return remove_orders(candidates)


def extract_cancellation(reply: str, sender: str) -> Optional[dict]:
    """
    Look for an ORDER_CANCELLED:{...} marker in the GPT reply.

    Returns {"phone": ..., "order_id": ..., "item": ...} describing WHICH order to
    cancel, or None if there's no marker. order_id/item may be None when the
    customer has only one order and didn't need to say which.
    """
    match = re.search(r"ORDER_CANCELLED:\s*(\{.*?\})", reply, re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        logger.warning("Found ORDER_CANCELLED marker but JSON did not parse: %s", match.group(1))
        payload = {}

    phone = payload.get("phone")
    if not phone or phone == "TWILIO_NUMBER":
        phone = sender

    order_id = payload.get("order_id") or payload.get("id")
    item = payload.get("item") or payload.get("items")
    if isinstance(item, (list, tuple)):
        item = ", ".join(str(entry) for entry in item)

    return {
        "phone": phone,
        "order_id": str(order_id) if order_id else None,
        "item": str(item) if item else None,
    }


# Injected as a system message so the agent confirms the cancellation itself, in
# the customer's own language.
ORDER_CANCELLED_INSTRUCTION = """SYSTEM NOTICE (not from the customer): \
The customer's order was CANCELLED and removed. Cancelled: {summary}. \
Confirm to the customer that their order has been cancelled, and offer to help if they'd like to order again. \
Do NOT output any ORDER_CONFIRMED, ORDER_UPDATED or ORDER_CANCELLED marker in this reply. \
Follow the language rule: reply in the language of the customer's latest message."""

ORDER_CANCELLED_FALLBACK = (
    "Your order has been cancelled — {summary}. "
    "Let me know if you'd like to place a new order."
)

NOTHING_TO_CANCEL_INSTRUCTION = """SYSTEM NOTICE (not from the customer): \
The customer asked to cancel, but they have NO active order to cancel. \
Tell them politely that you couldn't find an active order for them, and offer to place a new one. \
Do NOT output any marker in this reply. \
Follow the language rule: reply in the language of the customer's latest message."""

NOTHING_TO_CANCEL_FALLBACK = (
    "I couldn't find an active order to cancel for you. "
    "Would you like to place a new order?"
)

AMBIGUOUS_CANCEL_INSTRUCTION = """SYSTEM NOTICE (not from the customer): \
The customer asked to cancel, but they have MORE THAN ONE active order and it is not clear which one they mean. \
NOTHING has been cancelled yet. Their current orders are: {options}. \
List these orders back to the customer and ask WHICH ONE they want to cancel. \
Do NOT cancel anything and do NOT output any marker in this reply. \
Follow the language rule: reply in the language of the customer's latest message."""

AMBIGUOUS_CANCEL_FALLBACK = (
    "You have more than one active order: {options}. "
    "Which one would you like to cancel?"
)


def build_agent_notice_reply(sender: str, notice: str, fallback: str) -> str:
    """
    Ask the agent to phrase a system notice for the customer, in their language.

    Shared by the cancellation replies; mirrors the update/duplicate helpers by
    swapping the marker-bearing reply out of history for the phrased one, with a
    canned fallback if GPT is unavailable or the call fails.
    """
    history = get_history(sender)
    if len(history) > 1 and history[-1].get("role") == "assistant":
        history.pop()

    reply = ""
    if openai_client is not None:
        history.append({"role": "system", "content": notice})
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=history,
                temperature=0.6,
            )
            reply = completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — fall back to a canned message
            logger.exception("Notice reply generation failed: %s", exc)
        finally:
            history.pop()

    reply = clean_reply_for_user(reply) if reply.strip() else fallback
    history.append({"role": "assistant", "content": reply})
    trim_history(history)
    return reply


def handle_order_cancellation(
    reply: str, sender: str, channel: str
) -> tuple[Optional[str], Optional[list[dict]]]:
    """
    If the reply carries an ORDER_CANCELLED marker, delete the customer's
    order(s) and return a reply confirming it.

    The agent only emits the marker after the customer has confirmed (see the
    cancellation rules in SYSTEM_PROMPT), so reaching here means confirmation
    already happened.

    Returns (reply_to_send, cancelled_orders); (None, None) when there's no
    marker so the caller falls through to the create/update paths untouched.

    Only the order the customer actually named is cancelled. If they have several
    orders and it's unclear which one they mean, nothing is deleted and the agent
    asks them to pick.
    """
    target = extract_cancellation(reply, sender)
    if target is None:
        return None, None

    # Only ever cancel the sender's OWN orders. The phone in the marker is
    # ignored for deletion: trusting it would let a customer (or a confused
    # model) cancel somebody else's order. The DELETE endpoint is the way to
    # cancel by an arbitrary phone number.
    if normalize_phone(target["phone"]) != normalize_phone(sender):
        logger.warning(
            "ORDER_CANCELLED marker named %r but sender is %r — cancelling the sender's own order only.",
            target["phone"],
            sender,
        )

    candidates = orders_for_phone(sender)
    if not candidates:
        logger.info("Cancellation (%s) from %s but no active order found.", channel, sender)
        return (
            build_agent_notice_reply(
                sender, NOTHING_TO_CANCEL_INSTRUCTION, NOTHING_TO_CANCEL_FALLBACK
            ),
            [],
        )

    if len(candidates) == 1:
        # Only one order — no ambiguity about which one they mean.
        selected = candidates
    else:
        selected = select_orders_to_cancel(
            candidates, order_id=target["order_id"], item=target["item"]
        )
        if len(selected) != 1:
            # Either nothing matched, or the description fits several orders.
            # Ask rather than guess — deleting the wrong order isn't recoverable.
            options = "; ".join(describe_order(order) for order in candidates)
            logger.info(
                "Ambiguous cancellation (%s) from %s: %d candidate orders, marker item=%r id=%r — asking which.",
                channel,
                sender,
                len(candidates),
                target["item"],
                target["order_id"],
            )
            return (
                build_agent_notice_reply(
                    sender,
                    AMBIGUOUS_CANCEL_INSTRUCTION.format(options=options),
                    AMBIGUOUS_CANCEL_FALLBACK.format(options=options),
                ),
                [],
            )

    removed = remove_orders(selected)

    summary = "; ".join(describe_order(order) for order in removed)
    logger.info(
        "Cancelled %d order(s) (%s) from %s: %s",
        len(removed),
        channel,
        sender,
        [order.get("id") for order in removed],
    )
    return (
        build_agent_notice_reply(
            sender,
            ORDER_CANCELLED_INSTRUCTION.format(summary=summary),
            ORDER_CANCELLED_FALLBACK.format(summary=summary),
        ),
        removed,
    )


def handle_confirmed_order(
    reply: str, sender: str, profile_name: str, channel: str
) -> tuple[str, Optional[dict], Optional[dict]]:
    """
    Route a GPT reply that carries an order marker.

    ORDER_CANCELLED -> the customer already confirmed the cancellation, so their
    order is deleted and the reply confirms it.

    ORDER_UPDATED -> the customer's existing order (matched by phone) is updated
    in place and the reply confirms an update, not a new order.

    ORDER_CONFIRMED -> enrich and store a new order, unless an identical one
    (same phone + same items) already exists, in which case nothing is stored and
    the reply asks whether they want to modify it, cancel it, or if it was a
    mistake.

    Returns (reply_to_send, created_order, duplicate_order).
    Shared by every channel (webhook text, webhook voice, browser simulator).
    """
    cancel_reply, cancelled = handle_order_cancellation(reply, sender, channel)
    if cancelled is not None:
        return cancel_reply, None, None

    update_reply, updated = handle_order_update(reply, sender, channel)
    if updated is not None:
        return update_reply, None, None

    order = extract_order(reply, sender)
    if not order:
        return reply, None, None
    if profile_name and (not order.get("customer_name") or order["customer_name"] in ("", "...")):
        order["customer_name"] = profile_name

    duplicate = find_duplicate_order(order)
    if duplicate:
        logger.info(
            "Duplicate order suppressed (%s) from %s — matches existing order %s: %s",
            channel,
            sender,
            duplicate.get("id"),
            order,
        )
        return build_duplicate_order_reply(sender, duplicate), None, duplicate

    orders.append(order)
    logger.info("Order confirmed (%s) from %s: %s", channel, sender, order)
    return reply, order, None


def clean_reply_for_user(reply: str) -> str:
    """
    Strip the ORDER_CONFIRMED:{...} machine marker out of the message that the
    customer actually sees, so they get a clean human-readable confirmation.
    """
    cleaned = re.sub(
        r"ORDER_(?:CONFIRMED|UPDATED):\s*\{.*\}", "", reply, flags=re.DOTALL
    )
    cleaned = re.sub(
        r"ORDER_CANCELLED:\s*\{.*?\}", "", cleaned, flags=re.DOTALL
    ).strip()
    return cleaned or "Your order is confirmed! ✅"


# ---------------------------------------------------------------------------
# Voice note pipeline: download -> Deepgram -> (GPT) -> OpenAI TTS -> serve
# ---------------------------------------------------------------------------

async def download_twilio_media(media_url: str) -> bytes:
    """Download media (e.g. a WhatsApp voice note) from Twilio with auth."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            media_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.content


def normalize_deepgram_content_type(content_type: str) -> str:
    """
    Deepgram detects the audio container from the Content-Type header, but it
    expects a bare container mimetype (e.g. "audio/webm"). Browser MediaRecorder
    reports types like "audio/webm;codecs=opus" — the ";codecs=..." suffix can
    stop Deepgram from decoding the file, yielding an empty transcript. Strip the
    parameters and fall back to audio/webm (what the browser simulator sends).
    """
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    return ct or "audio/webm"


async def transcribe_with_deepgram(audio_bytes: bytes, content_type: str) -> str:
    """Send audio bytes to Deepgram and return the transcript text."""
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured.")

    deepgram_content_type = normalize_deepgram_content_type(content_type)

    url = "https://api.deepgram.com/v1/listen"
    params = {"model": "nova-2", "smart_format": "true", "detect_language": "true"}
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": deepgram_content_type,
    }
    logger.info(
        "Sending %d bytes to Deepgram (Content-Type: %s)",
        len(audio_bytes),
        deepgram_content_type,
    )
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, params=params, headers=headers, content=audio_bytes)
        resp.raise_for_status()
        data = resp.json()

    try:
        channel = data["results"]["channels"][0]
        transcript = channel["alternatives"][0]["transcript"]
    except (KeyError, IndexError):
        logger.warning("Unexpected Deepgram response shape: %s", data)
        return ""

    # Guard against spurious language detection (e.g. Deepgram occasionally tags
    # short/noisy audio as Chinese/Japanese, which then makes GPT reply in that
    # language). We only support English + Urdu/Hindi here — anything else is
    # almost certainly a misdetection, so log it and treat the audio as English.
    detected = (channel.get("detected_language") or "").lower()
    if detected and detected[:2] not in ALLOWED_TRANSCRIPTION_LANGUAGES:
        logger.warning(
            "Deepgram detected unsupported language %r — treating as English. Transcript: %r",
            detected,
            transcript,
        )

    return transcript


async def synthesize_bytes_with_openai(text: str) -> bytes:
    """Convert text to speech with OpenAI TTS and return the raw mp3 bytes."""
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    # The SDK's TTS call is synchronous; run it off the event loop.
    def _create() -> bytes:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=OPENAI_TTS_VOICE,
            input=text,
        )
        return response.content

    return await asyncio.to_thread(_create)


async def synthesize_with_openai(text: str) -> str:
    """
    Convert text to speech with OpenAI TTS, save as an .mp3 in AUDIO_DIR,
    and return the local file name.
    """
    audio = await synthesize_bytes_with_openai(text)
    file_name = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join(AUDIO_DIR, file_name)
    with open(file_path, "wb") as f:
        f.write(audio)
    return file_name


def send_whatsapp_text(to: str, body: str) -> None:
    """Send a plain text WhatsApp message via Twilio."""
    if twilio_client is None:
        logger.error("Twilio client not configured — cannot send message.")
        return
    twilio_client.messages.create(from_=TWILIO_WHATSAPP_NUMBER, to=to, body=body)


def send_whatsapp_voice(to: str, audio_file_name: str, caption: str = "") -> None:
    """
    Send a voice note (audio file) back via Twilio. Twilio fetches the media
    from a publicly reachable URL, so PUBLIC_BASE_URL must be set.
    """
    if twilio_client is None:
        logger.error("Twilio client not configured — cannot send voice note.")
        return
    if not PUBLIC_BASE_URL:
        logger.warning("PUBLIC_BASE_URL not set — falling back to text reply.")
        send_whatsapp_text(to, caption or "(voice reply unavailable)")
        return

    media_url = f"{PUBLIC_BASE_URL}/audio/{audio_file_name}"
    twilio_client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        to=to,
        body=caption or None,
        media_url=[media_url],
    )


# ---------------------------------------------------------------------------
# Request models for the browser-based testing simulator
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    sender: str


class ChatVoiceRequest(BaseModel):
    # Base64-encoded audio. Optionally a data URL (e.g. "data:audio/webm;base64,....").
    audio: str
    sender: str
    content_type: str = "audio/webm"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Simple health check with a snapshot of in-memory state."""
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "active_conversations": len(conversations),
        "total_orders": len(orders),
    }


@app.get("/orders")
async def list_orders():
    """Return all confirmed orders held in memory."""
    return {"count": len(orders), "orders": orders}


@app.delete("/orders/{phone}")
async def cancel_orders(phone: str, order_id: Optional[str] = None, item: Optional[str] = None):
    """
    Cancel (delete) orders belonging to a phone number.

    Used by the cancellation flow once the customer has confirmed. The phone may
    be given in any format — "+923001234567", "03001234567" or a full
    "whatsapp:+92..." sender id all match the same customer.

    Pass ?order_id=... or ?item=... to cancel one specific order; with neither,
    all of that customer's orders are cancelled.

    Returns 404 when there's nothing to cancel so the caller can tell the
    difference between "deleted" and "no such order".
    """
    removed = delete_orders_for_phone(phone, order_id=order_id, item=item)
    if not removed:
        return JSONResponse(
            status_code=404,
            content={
                "error": "no matching active order found for this phone number",
                "phone": phone,
                "order_id": order_id,
                "item": item,
            },
        )

    logger.info("Cancelled %d order(s) via DELETE /orders/%s", len(removed), phone)
    return {
        "cancelled": len(removed),
        "phone": phone,
        "orders": removed,
        "remaining": len(orders),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Browser simulator: same agent logic as the webhook text branch, but returns
    the reply as JSON instead of sending it through Twilio.
    """
    sender = (req.sender or "").strip()
    message = (req.message or "").strip()
    if not sender:
        return JSONResponse(status_code=400, content={"error": "missing sender"})
    if not message:
        return JSONResponse(status_code=400, content={"error": "missing message"})

    try:
        reply = run_gpt(sender, message)
    except Exception as exc:  # noqa: BLE001 — demo: report failure to the caller
        logger.exception("GPT call failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": "AI processing failed"})

    reply, _order, _duplicate = handle_confirmed_order(reply, sender, profile_name="", channel="chat")
    return {"reply": clean_reply_for_user(reply)}


@app.post("/chat-voice")
async def chat_voice(req: ChatVoiceRequest):
    """
    Browser simulator voice pipeline: base64 audio in -> Deepgram -> GPT ->
    OpenAI TTS -> base64 audio out. Same agent logic as the webhook voice branch,
    minus Twilio.
    """
    sender = (req.sender or "").strip()
    if not sender:
        return JSONResponse(status_code=400, content={"error": "missing sender"})

    # Accept either a raw base64 string or a "data:...;base64,..." data URL.
    raw_b64 = req.audio or ""
    if "," in raw_b64 and raw_b64.strip().startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]
    try:
        audio_bytes = base64.b64decode(raw_b64)
    except (ValueError, TypeError):
        return JSONResponse(status_code=400, content={"error": "invalid base64 audio"})
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "empty audio"})

    try:
        transcript = await transcribe_with_deepgram(audio_bytes, req.content_type)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": "transcription failed"})

    logger.info("Transcript (chat-voice) from %s: %r", sender, transcript)
    if not transcript.strip():
        # Deepgram heard nothing intelligible — return a friendly text-only reply
        # (audio: null) instead of erroring so the simulator can show a bubble.
        return {
            "reply": "Sorry, I couldn't hear that clearly. Could you try again?",
            "audio": None,
        }

    try:
        reply = run_gpt(sender, transcript)
    except Exception as exc:  # noqa: BLE001
        logger.exception("GPT call failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": "AI processing failed"})

    reply, _order, _duplicate = handle_confirmed_order(reply, sender, profile_name="", channel="chat-voice")
    spoken = clean_reply_for_user(reply)

    try:
        audio_out = await synthesize_bytes_with_openai(spoken)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Speech synthesis failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": "speech synthesis failed"})

    return {
        "transcript": transcript,
        "reply": spoken,
        "audio": base64.b64encode(audio_out).decode("ascii"),
        "audio_content_type": "audio/mpeg",
    }


@app.get("/audio/{file_name}")
async def get_audio(file_name: str):
    """Serve a generated reply audio file (so Twilio can fetch it)."""
    # Guard against path traversal — only allow plain file names.
    safe_name = os.path.basename(file_name)
    file_path = os.path.join(AUDIO_DIR, safe_name)
    if not os.path.isfile(file_path):
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(file_path, media_type="audio/mpeg")


@app.post("/webhook")
async def webhook(request: Request):
    """
    Twilio WhatsApp webhook. Twilio posts application/x-www-form-urlencoded.

    Handles both:
      - text messages (Body)
      - voice notes    (MediaUrl0 + MediaContentType0)
    """
    form = await request.form()
    sender = form.get("From", "")            # e.g. "whatsapp:+92300..."
    profile_name = form.get("ProfileName", "")
    body = (form.get("Body") or "").strip()
    num_media = int(form.get("NumMedia", 0) or 0)

    logger.info("Inbound from %s (%s) media=%d body=%r", sender, profile_name, num_media, body)

    if not sender:
        return JSONResponse(status_code=400, content={"error": "missing sender"})

    # ----- Voice note branch -------------------------------------------------
    is_voice = num_media > 0 and (form.get("MediaContentType0", "")).startswith("audio")
    if is_voice:
        try:
            await handle_voice_note(sender, profile_name, form)
        except Exception as exc:  # noqa: BLE001 — demo: report any failure to user
            logger.exception("Voice note handling failed: %s", exc)
            send_whatsapp_text(sender, "Sorry, I couldn't process that voice note. Please try again or type your order.")
        return Response(status_code=200)

    # ----- Text branch -------------------------------------------------------
    if not body:
        send_whatsapp_text(sender, "Please send a message describing what you'd like to order. 🍔")
        return Response(status_code=200)

    try:
        reply = run_gpt(sender, body)
    except Exception as exc:  # noqa: BLE001
        logger.exception("GPT call failed: %s", exc)
        send_whatsapp_text(sender, "Sorry, something went wrong. Please try again in a moment.")
        return Response(status_code=200)

    _finalize_text_reply(sender, profile_name, reply)
    return Response(status_code=200)


async def handle_voice_note(sender: str, profile_name: str, form) -> None:
    """Full voice pipeline: download -> transcribe -> GPT -> TTS -> send back."""
    media_url = form.get("MediaUrl0")
    content_type = form.get("MediaContentType0", "audio/ogg")

    audio_bytes = await download_twilio_media(media_url)
    transcript = await transcribe_with_deepgram(audio_bytes, content_type)
    logger.info("Transcript from %s: %r", sender, transcript)

    if not transcript:
        send_whatsapp_text(sender, "I couldn't hear anything in that voice note. Could you try again?")
        return

    reply = run_gpt(sender, transcript)

    # Persist any confirmed order (same logic as text path).
    reply, _order, _duplicate = handle_confirmed_order(reply, sender, profile_name, channel="voice")

    spoken = clean_reply_for_user(reply)

    # Generate audio reply and send it back as a voice note.
    audio_file = await synthesize_with_openai(spoken)
    send_whatsapp_voice(sender, audio_file, caption=spoken)


def _finalize_text_reply(sender: str, profile_name: str, reply: str) -> None:
    """Persist any confirmed order, then send the cleaned reply to the user."""
    reply, _order, _duplicate = handle_confirmed_order(reply, sender, profile_name, channel="text")
    send_whatsapp_text(sender, clean_reply_for_user(reply))


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
