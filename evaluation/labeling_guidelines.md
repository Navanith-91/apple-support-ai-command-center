# Apple Support Golden Set Human Annotation Guidelines

This document provides standardized, objective criteria for human annotators creating the 200-sample Golden Evaluation Set for **Apple Support** (`@AppleSupport`).

---

## 1. Intent Taxonomy (9 Canonical Classes)

Annotators must assign exactly **one primary intent** from the 9 classes below.

---

### 1. `battery_and_charging`
* **Definition**: Inquiries and complaints regarding device battery drain, rapid discharging, overheating during charging, faulty cables/chargers, battery health percentage drops, or failure to power on due to battery issues.
* **What Belongs**:
  - iPhone/iPad/Mac battery draining quickly after normal use.
  - Device getting hot while charging or playing media.
  - Lightning/USB-C charging port not recognizing cable or charging intermittently.
  - Maximum battery capacity dropping unexpectedly in iOS Settings.
* **What Does NOT Belong**:
  - Battery drain specifically caused by an iOS software update glitch (assign to `ios_software_update_bugs` if update is the explicit cause).
  - General physical screen or speaker hardware failure (assign to `hardware_audio_and_screen`).
* **Representative Examples**:
  1. *"My iPhone 7 battery is draining from 100% to 15% in less than three hours after normal use. Why is this happening?"*
  2. *"My phone only charges when I hold the cable at a specific angle. Is my charging port broken?"*
  3. *"Battery health dropped from 98% to 84% in two weeks. Is this normal?"*
* **Boundary Case**: If the user says *"My battery started draining immediately after updating to iOS 11.0.3"*, classify as `ios_software_update_bugs` because the update is the focal operational event.

---

### 2. `app_store_billing_and_subscriptions`
* **Definition**: Financial transactions, unrecognized credit card charges from Apple/iTunes, refund requests for apps/in-app purchases/music, subscription cancellation issues, or payment method declines.
* **What Belongs**:
  - Unrecognized charges on credit/debit card from `itunes.com/bill` or Apple Services.
  - Accidental in-app purchases made by children or subscriptions renewing without consent.
  - Payment method declined in App Store or Apple Pay setup.
  - Inquiries on how to request an App Store refund.
* **What Does NOT Belong**:
  - Physical Apple Store hardware purchases, retail pickups, or trade-in kits (assign to `order_shipping_and_trade_in`).
  - Forgotten Apple ID passwords or two-factor authentication (assign to `apple_id_and_icloud`).
* **Representative Examples**:
  1. *"I see an unauthorized charge of $89.99 from itunes.com/bill on my credit card that I never made. I demand an immediate refund!"*
  2. *"How do I cancel my Apple Music trial before it charges my card next week?"*
  3. *"My card is declined on the App Store even though I have sufficient funds."*
* **Boundary Case**: If the customer says *"I forgot my password and now my card got charged for an app"*, the primary financial harm takes precedence (`app_store_billing_and_subscriptions`) unless the core blocker is pure account access.

---

### 3. `ios_software_update_bugs`
* **Definition**: Operational bugs, app crashes, system UI freezes, keyboard lag, Bluetooth dropouts, Wi-Fi connectivity drops, or boot loops occurring after or during an iOS/macOS update.
* **What Belongs**:
  - Device stuck on the Apple logo or progress bar during an update.
  - Built-in apps (Camera, Music, Messages) crashing repeatedly on a new iOS build.
  - Keyboard lag, screen stutter, or Bluetooth disconnects immediately following a software update.
  - General OS performance degradation attributed to recent software releases.
* **What Does NOT Belong**:
  - Physical screen cracks, water damage, or damaged speaker grilles (assign to `hardware_audio_and_screen`).
  - Questions on *how* to update or how to backup before updating (assign to `general_how_to_and_features`).
* **Representative Examples**:
  1. *"Ever since updating to iOS 11.1, my calculator app keeps freezing and the keyboard lags."*
  2. *"My iPhone is stuck on the Apple logo boot loop after attempting the OTA update."*
  3. *"Wi-Fi keeps disconnecting every 5 minutes on my MacBook after the latest macOS patch."*
* **Boundary Case**: If the user asks *"Is iOS 11 available for iPhone 6?"*, classify as `general_how_to_and_features` (informational), whereas *"iOS 11 broke my Wi-Fi"* is `ios_software_update_bugs`.

---

### 4. `general_complaint_and_escalation`
* **Definition**: Expressions of severe customer dissatisfaction, threats to switch to competitor brands (Android/Samsung), negative experiences with retail Genius Bar staff, or demands for a manager/supervisor without a specific isolated technical issue.
* **What Belongs**:
  - Ineffective previous support interactions (*"I've contacted you 4 times and nobody fixed it"*).
  - Explicit demands to speak with a supervisor, manager, or corporate representative.
  - General outrage or legal threats (*"I'm going to sue Apple for this"*).
  - Venting about overall Apple quality without describing a concrete troubleshooting request.
* **What Does NOT Belong**:
  - A technical bug reported with mild frustration (classify by the technical intent, e.g. `battery_and_charging`).
  - Financial disputes with specific transaction amounts (classify as `app_store_billing_and_subscriptions`).
* **Representative Examples**:
  1. *"Your customer support is completely useless! I've been waiting for two weeks and nobody resolved my case. Get me a supervisor right now!"*
  2. *"Apple's quality has gone down the drain. I am switching all my devices to Android tomorrow."*
  3. *"The staff at the Covent Garden Apple Store were extremely rude to me today. Who do I complain to?"*
* **Boundary Case**: If a customer reports a broken screen AND demands a supervisor, prioritize `general_complaint_and_escalation` if the dominant request is escalation/dissatisfaction.

---

### 5. `order_shipping_and_trade_in`
* **Definition**: Logistics, delivery tracking, delayed shipping, Apple Online Store order status, retail store pickup scheduling, or trade-in kit return status.
* **What Belongs**:
  - Tracking numbers for newly ordered iPhones, iPads, or MacBooks.
  - Delayed shipments or lost packages from courier partners (UPS, FedEx, DHL).
  - Trade-in kit status, return box requests, or trade-in valuation disputes.
  - Apple Retail Store order pickup time slot inquiries.
* **What Does NOT Belong**:
  - Digital App Store charges or subscription receipts (assign to `app_store_billing_and_subscriptions`).
  - Technical warranty repairs at a repair center (assign to appropriate hardware intent).
* **Representative Examples**:
  1. *"My iPhone X order was supposed to arrive yesterday via UPS but tracking still says label created."*
  2. *"When will my trade-in kit arrive so I can send back my old iPhone 7?"*
  3. *"Can I change my in-store pickup location for order #W12345678?"*

---

### 6. `hardware_audio_and_screen`
* **Definition**: Physical defects, cracked glass, non-responsive touch digitizers, broken speaker/earpiece, microphone static, camera blur/shaking, Face ID/Touch ID sensor failures, or liquid ingress.
* **What Belongs**:
  - Physical screen damage, vertical green lines on OLED screens, dead pixels.
  - Earpiece sound distorted or callers cannot hear user through the microphone.
  - Camera vibrating or unable to focus on iPhone.
  - Water/liquid contact warnings and physical button unresponsiveness.
* **What Does NOT Belong**:
  - Battery degradation with no physical bulging (assign to `battery_and_charging`).
  - Software keyboard lag with functional touch screen (assign to `ios_software_update_bugs`).
* **Representative Examples**:
  1. *"My iPhone screen has vertical green lines and the bottom half doesn't respond to touch."*
  2. *"People can't hear me on phone calls unless I turn on speakerphone. Is the mic broken?"*
  3. *"My camera lens makes a buzzing noise and won't focus."*

---

### 7. `apple_id_and_icloud`
* **Definition**: Account authentication, locked Apple IDs, forgotten passwords, two-factor authentication (2FA) verification code delivery failures, iCloud storage sync errors, and Activation Lock inquiries.
* **What Belongs**:
  - *"Apple ID locked for security reasons."*
  - Password reset emails not arriving or recovery keys lost.
  - Unable to receive 2FA verification SMS on trusted device.
  - iCloud Photos / Drive not syncing across devices or storage quota confusion.
  - Activation Lock on a pre-owned device.
* **What Does NOT Belong**:
  - Unrecognized App Store credit card charge with known password (assign to `app_store_billing_and_subscriptions`).
  - General how-to on changing ringtones or wallpaper (assign to `general_how_to_and_features`).
* **Representative Examples**:
  1. *"My Apple ID has been locked for security reasons and I can't receive the two-factor code on my trusted phone number."*
  2. *"I forgot my Apple ID password and the recovery email is an old inbox I don't have access to."*
  3. *"My iCloud photos stopped syncing between my Mac and my iPhone."*

---

### 8. `general_how_to_and_features`
* **Definition**: Informational requests and user instructions on how to use standard iOS/macOS features, configure settings, transfer data, pair Bluetooth accessories, use AirDrop/CarPlay, or check feature compatibility.
* **What Belongs**:
  - How to transfer data/photos from an older iPhone to a new device.
  - How to pair AirPods or third-party Bluetooth headphones.
  - How to enable Do Not Disturb, Dark Mode, Screen Time, or AirDrop.
  - Feature capability inquiries (*"Does iPhone 8 support fast charging?"*).
* **What Does NOT Belong**:
  - System crashes or bugs occurring during feature usage (assign to `ios_software_update_bugs`).
  - Account security setup issues like 2FA lockout (assign to `apple_id_and_icloud`).
* **Representative Examples**:
  1. *"How do I transfer all my photos and contacts from an older iPhone to my new iPhone 8 using Quick Start?"*
  2. *"How can I enable Do Not Disturb while driving on iOS 11?"*
  3. *"Can I connect two pairs of Bluetooth headphones to my iPad at the same time?"*

---

### 9. `other`
* **Definition**: Unrelated queries outside Apple Support's remit, pure spam, social banter, emojis without text, single-word gibberish, or customer messages so ambiguous that no domain intent is discernable.
* **What Belongs**:
  - *"hey apple"* / *"cool"* / *"😀😀😀"*
  - Third-party non-Apple software with zero Apple involvement (*"how do I install Windows XP on my Dell laptop?"*).
  - Totally ambiguous fragments (*"it broke ????"*).
* **Representative Examples**:
  1. *"hey what's up"*
  2. *"great job apple love you guys"*
  3. *"???"*

---

## 2. Decision Guidelines (AUTO_HANDLE vs. ESCALATE_TO_HUMAN)

We operate on a **Safety-First Principle**:
> **False Auto-Handling Rate (FAHR) must be minimized above all else.**
> It is far better to unnecessarily escalate a routine issue to a human specialist than to inappropriately auto-respond to an account takeover, unauthorized charge, or legal dispute.

### Assign `ESCALATE_TO_HUMAN` when ANY of the following apply:
1. **Account Security & Identity**:
   - Locked Apple IDs, lost 2FA verification devices, password recovery loops.
2. **Financial Transactions & Fraud**:
   - Unrecognized credit card charges, refund requests, billing disputes, unauthorized in-app purchases.
3. **Severe Customer Frustration / Escalation Signals**:
   - Customer demanding a manager/supervisor, expressing extreme anger, or threatening legal action/switching brands.
4. **Physical Safety & Hardware Hazards**:
   - Bulging batteries, smoke, thermal events, or hardware replacement scheduling.
5. **High Ambiguity / Unanswerable Context**:
   - Messages where automated troubleshooting advice could mislead the user or where context is critically missing.

### Assign `AUTO_HANDLE` when ALL of the following apply:
1. The message is a routine informational, how-to, or standard software troubleshooting issue.
2. An official, safe troubleshooting step (e.g. restart device, check settings menu, update app, reset network settings) exists in standard Apple knowledge base.
3. No personal account credentials, credit card details, or individualized backend actions are required.
4. The customer is not expressing extreme anger or demanding a human supervisor.

---

## 3. Reason Requirement

Every annotation MUST include a non-empty `gold_reason` explaining why the decision is appropriate. Examples:
- *"Routine how-to query regarding Quick Start data migration resolvable via standard documentation."*
- *"Unauthorized financial charge requires human agent verification and secure billing portal referral."*
- *"Apple ID security lockout requires authenticated identity recovery flow."*
- *"Customer is expressing severe frustration and demanding a supervisor."*
