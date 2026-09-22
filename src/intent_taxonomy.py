"""Intent taxonomy definition, inclusion/exclusion criteria, and disambiguation rules."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class IntentSpec:
    id: str
    name: str
    description: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    representative_examples: List[str]
    ambiguity_notes: str
    keywords: List[str] = field(default_factory=list)


INTENT_TAXONOMY: Dict[str, IntentSpec] = {
    "battery_and_charging": IntentSpec(
        id="battery_and_charging",
        name="Battery & Charging Issues",
        description="Problems regarding iPhone/iPad battery drain, fast discharge, heating while charging, charger/cable defects, or battery health drops.",
        inclusion_criteria=[
            "Mentions battery percentage drops, rapid battery drain, shutting down with remaining percentage.",
            "Mentions phone getting hot/overheating during charge or normal use.",
            "Mentions charger not working, lightning cable issues, slow charging.",
            "Inquires about battery health percentage or battery replacement."
        ],
        exclusion_criteria=[
            "Complaints about iOS update bug where battery is only mentioned as one minor symptom (classify as ios_software_update_bugs if focused on update).",
            "General hardware damage like broken screen (classify as hardware_audio_and_screen)."
        ],
        representative_examples=[
            "My iPhone 7 battery is draining from 100% to 20% in two hours after doing nothing.",
            "Why is my phone getting extremely hot while charging on iOS 11?",
            "My charger stopped working and says this accessory may not be supported."
        ],
        ambiguity_notes="If customer mentions 'battery drain after updating to iOS 11', prefer ios_software_update_bugs if the complaint focuses on the update itself, but battery_and_charging if asking specifically for battery diagnosis/replacement.",
        keywords=["battery", "drain", "draining", "charge", "charging", "charger", "overheating", "hot", "battery life", "die quickly", "percent", "cable"]
    ),
    "ios_software_update_bugs": IntentSpec(
        id="ios_software_update_bugs",
        name="iOS & Software Update Bugs",
        description="Bugs, UI glitches, app freezes, keyboard lag, Bluetooth/WiFi connectivity dropouts, or OS crashes caused by an iOS or macOS update.",
        inclusion_criteria=[
            "Explicit mention of iOS 11, iOS 12, iOS update, update broke my phone.",
            "Keyboard autocorrect bugs (e.g. the famous 'I' to 'A [?]' bug on iOS 11).",
            "Apps crashing, freezing, screen lagging or stuttering after software updates.",
            "System UI glitches (Control Center, notifications, lock screen)."
        ],
        exclusion_criteria=[
            "Physical broken screen or physical speaker distortion (classify as hardware_audio_and_screen).",
            "iCloud sync failure or Apple ID lock (classify as apple_id_and_icloud)."
        ],
        representative_examples=[
            "Ever since updating to iOS 11.1, whenever I type the letter 'I' it changes to a weird symbol.",
            "My iPhone keeps freezing and rebooting itself constantly since the last update.",
            "Control Center is completely unresponsive and lagging on iOS 11."
        ],
        ambiguity_notes="High prevalence in Twitter dataset due to iOS 11 release period. Distinguish from general hardware faults by looking for update references or software-specific symptoms.",
        keywords=["ios", "update", "updated", "ios 11", "software", "bug", "glitch", "freeze", "freezing", "lag", "crash", "reboot", "autocorrect", "keyboard"]
    ),
    "apple_id_and_icloud": IntentSpec(
        id="apple_id_and_icloud",
        name="Apple ID & iCloud Access",
        description="Account security, locked Apple ID, two-factor authentication, forgotten passwords, verification codes, and iCloud storage syncing issues.",
        inclusion_criteria=[
            "Apple ID locked for security reasons, verification code not received.",
            "Cannot log in, forgot password, recovery key / account recovery process.",
            "iCloud photos not backing up, iCloud storage full warnings, sync errors.",
            "Two-factor authentication (2FA) loop or trusted device issues."
        ],
        exclusion_criteria=[
            "App Store billing or credit card charges (classify as app_store_billing_and_subscriptions).",
            "General hardware repair inquiry (classify as hardware_audio_and_screen)."
        ],
        representative_examples=[
            "My Apple ID has been locked for security reasons and I can't receive the verification code on my trusted number.",
            "How do I recover my iCloud password if I no longer have access to my old email?",
            "My photos are not uploading to iCloud even though I bought the 200GB storage plan."
        ],
        ambiguity_notes="High-risk category. Involves identity and access. Any credential recovery or locked account must escalate to human or official secure link.",
        keywords=["apple id", "icloud", "password", "locked", "verification code", "2fa", "two-factor", "account", "login", "recover", "sync", "storage", "security code"]
    ),
    "hardware_audio_and_screen": IntentSpec(
        id="hardware_audio_and_screen",
        name="Hardware, Screen & Audio Defects",
        description="Physical defects, cracked screen, unresponsive digitizer, microphone/speaker distortion, AirPods/headphone jack issues, or camera malfunctions.",
        inclusion_criteria=[
            "Cracked glass, OLED burn-in, lines on display, touchscreen dead zones.",
            "No sound from speaker, crackling noise on calls, microphone not working.",
            "AirPods not connecting, one AirPod quiet/dead, lightning headphone adapter issues.",
            "Camera lens blur, autofocus clicking, flash not working."
        ],
        exclusion_criteria=[
            "Software keyboard lag without physical screen defect (classify as ios_software_update_bugs).",
            "Battery degradation without physical swelling (classify as battery_and_charging)."
        ],
        representative_examples=[
            "My iPhone screen has a green vertical line running down the right side.",
            "The top speaker on my iPhone 8 has static and crackling noise during calls.",
            "My left AirPod won't charge or connect when taken out of the case."
        ],
        ambiguity_notes="Check if screen issue is software freezing (software bug) or physical hardware/touch failure.",
        keywords=["screen", "display", "cracked", "touch", "speaker", "audio", "microphone", "sound", "airpods", "headphones", "camera", "hardware", "static", "crackling"]
    ),
    "app_store_billing_and_subscriptions": IntentSpec(
        id="app_store_billing_and_subscriptions",
        name="App Store, Billing & Subscriptions",
        description="Charges on bank/credit card statements, in-app purchase disputes, refund requests, Apple Music / iCloud subscription cancellations, payment method declined.",
        inclusion_criteria=[
            "Inquiring about unknown charge from iTunes/App Store.",
            "Requesting refund for accidental app purchase or in-app purchase.",
            "Canceling recurring subscription (Apple Music, Apple Arcade, third-party app).",
            "Payment method declined in App Store or family sharing billing."
        ],
        exclusion_criteria=[
            "Purchases of physical hardware from Apple Online Store (classify as order_shipping_and_trade_in).",
            "iCloud storage sync problem without billing question (classify as apple_id_and_icloud)."
        ],
        representative_examples=[
            "I was charged $9.99 on my credit card by itunes.com/bill and I didn't buy anything.",
            "How do I get a refund for an accidental purchase made by my kid in a game?",
            "I canceled my Apple Music subscription last month but was still charged today."
        ],
        ambiguity_notes="High-risk financial intent. Must always require human handling or official reportaproblem.apple.com authentication.",
        keywords=["refund", "charged", "billing", "charge", "subscription", "cancel subscription", "payment", "credit card", "itunes.com/bill", "in-app purchase", "receipt", "money"]
    ),
    "order_shipping_and_trade_in": IntentSpec(
        id="order_shipping_and_trade_in",
        name="Order, Shipping & Trade-In",
        description="Apple Online Store physical device orders, delivery delays, UPS/FedEx tracking, trade-in kit delivery, order cancellations, or in-store pickup.",
        inclusion_criteria=[
            "Tracking order number for iPhone, Mac, Apple Watch or accessories.",
            "Delayed delivery, courier delivery issues, lost package.",
            "Trade-in box not received, trade-in valuation status.",
            "Canceling or modifying an online hardware order before shipment."
        ],
        exclusion_criteria=[
            "App Store digital download issues (classify as app_store_billing_and_subscriptions).",
            "In-store repair appointment (classify as general_how_to_and_features or general_complaint_and_escalation)."
        ],
        representative_examples=[
            "My iPhone X preorder status has been stuck on 'Preparing to Ship' for three days.",
            "I never received the trade-in return kit for my old iPhone 6s.",
            "Can I change the delivery address on my Apple Store order W123456789?"
        ],
        ambiguity_notes="Distinguish digital App Store orders (billing) from physical hardware shipments (order_shipping).",
        keywords=["order", "shipping", "delivery", "track", "tracking", "trade-in", "package", "ups", "fedex", "preorder", "arriving", "shipment", "store pickup"]
    ),
    "general_how_to_and_features": IntentSpec(
        id="general_how_to_and_features",
        name="How-To & Feature Inquiries",
        description="Configuration guides, transferring data to a new device, setting up Apple Pay, pairing Bluetooth accessories, enabling Dark Mode, or feature how-tos.",
        inclusion_criteria=[
            "Asking how to perform a specific function or enable a feature in iOS/macOS.",
            "Data transfer questions (Quick Start, migration from Android/old iPhone).",
            "Questions about feature compatibility or specifications.",
            "Setting up Apple Pay, Do Not Disturb, AirDrop, Screen Time."
        ],
        exclusion_criteria=[
            "Reporting a bug where an existing feature is broken (classify as ios_software_update_bugs).",
            "Requesting refund or account unlock."
        ],
        representative_examples=[
            "How do I transfer my photos and contacts from an Android phone to iPhone 8?",
            "Can I use two pairs of AirPods with one iPhone at the same time?",
            "How do I turn off autocorrect completely in iOS?"
        ],
        ambiguity_notes="Safe to auto-handle when standard documentation / support articles provide exact steps.",
        keywords=["how do i", "how to", "how can i", "setup", "set up", "transfer", "airdrop", "apple pay", "bluetooth", "pair", "feature", "configure", "settings"]
    ),
    "general_complaint_and_escalation": IntentSpec(
        id="general_complaint_and_escalation",
        name="General Complaints & Dissatisfaction",
        description="Frustrated feedback, threats to switch brands, poor customer service experience in Apple Store / phone support, or demanding to speak to a manager.",
        inclusion_criteria=[
            "Customer venting extreme frustration without a clear actionable technical inquiry.",
            "Complaints about terrible customer service, unhelpful Genius Bar staff, or long wait times.",
            "Demanding a supervisor, manager, or immediate human escalation.",
            "Threatening to switch to Samsung/Android or file legal/regulatory complaints."
        ],
        exclusion_criteria=[
            "Specific technical issue with polite or neutral tone (classify under specific technical intent).",
            "Spam or nonsense text (classify as other)."
        ],
        representative_examples=[
            "Apple customer service has gone down the drain. Terrible experience at the store today!",
            "I have been waiting for 3 weeks and nobody is helping me. Get me a manager right now!",
            "Worst company ever, switching to Samsung tomorrow. Fix your broken trash."
        ],
        ambiguity_notes="Escalation signal. Even if minor technical details are mentioned, high anger and explicit demand for humans take precedence for brand protection.",
        keywords=["terrible", "worst", "horrible", "unacceptable", "supervisor", "manager", "human", "agent", "switch to samsung", "switching to android", "waste of money", "disgusted", "lawsuit"]
    ),
    "other": IntentSpec(
        id="other",
        name="Other / Unrelated",
        description="Incomprehensible messages, single-word greetings, emojis only, non-English text without translation, or inquiries outside Apple Support scope.",
        inclusion_criteria=[
            "Single-word or emoji messages (e.g. 'hi', 'apple pls', '???', '🔥🔥').",
            "Unrelated marketing queries, fan commentary, or random mentions.",
            "Incoherent or truncated messages lacking sufficient context to determine intent."
        ],
        exclusion_criteria=[
            "Any message with a determinable technical or support inquiry."
        ],
        representative_examples=[
            "Hello Apple!",
            "iphone x is fire 🔥🔥🔥",
            "help please ????"
        ],
        ambiguity_notes="Catch-all category for unclassifiable or zero-context input. Generally triggers clarification prompt or escalation.",
        keywords=["hi", "hello", "hey", "pls", "help", "test", "fire", "cool", "apple", "thanks"]
    )
}


def get_all_intent_ids() -> List[str]:
    """Returns list of valid intent IDs."""
    return list(INTENT_TAXONOMY.keys())


def get_intent_spec(intent_id: str) -> Optional[IntentSpec]:
    """Retrieves spec for a given intent ID."""
    return INTENT_TAXONOMY.get(intent_id)


def get_intent_descriptions_formatted() -> str:
    """Formats intent descriptions for prompt inclusion."""
    lines = []
    for k, v in INTENT_TAXONOMY.items():
        lines.append(f"- **{v.id}** ({v.name}): {v.description}")
        lines.append(f"  * Key indicators: {', '.join(v.keywords[:6])}")
    return "\n".join(lines)
