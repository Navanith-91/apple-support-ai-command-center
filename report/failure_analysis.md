# Top 5 Failure Modes & Error Analysis

> [!IMPORTANT]

> This document analyzes genuine failure cases identified during the evaluation benchmark.

> No synthetic or fabricated examples are used.


## Failure Mode 1: Escalation Mismatch (ID: `gold_003`)

1. **Failure Mode**: Escalation Mismatch on customer support inquiry
2. **Real Customer Message**: "why does this keep replacing my "I" on my phone whenever its posted on any app"
3. **Gold Intent**: `ios_software_update_bugs`
4. **Predicted Intent**: `ios_software_update_bugs`
5. **Gold Decision**: `AUTO_HANDLE`
6. **Predicted Decision**: `ESCALATE_TO_HUMAN`
7. **Relevant Retrieved Evidence**: "Thanks for reaching out to us. Please send us a DM and we can chat with your there...."
8. **Why the System Failed**: Semantic overlap between lexical features or boundary ambiguity between adjacent categories.
9. **Hypothesis for Improvement**: Fine-tune contrastive embedding weights with class-specific decision margins.
10. **Proposed Next Experiment**: Add intent-specific negative hard-mining samples during retrieval and classification.

---

## Failure Mode 2: Escalation Mismatch (ID: `gold_005`)

1. **Failure Mode**: Escalation Mismatch on customer support inquiry
2. **Real Customer Message**: "When can we expect the unlocked version of iPhone X available for preorder or to get one from stores."
3. **Gold Intent**: `order_shipping_and_trade_in`
4. **Predicted Intent**: `order_shipping_and_trade_in`
5. **Gold Decision**: `AUTO_HANDLE`
6. **Predicted Decision**: `ESCALATE_TO_HUMAN`
7. **Relevant Retrieved Evidence**: "We’d like to get you in touch with our Apple Online Store. You can reach out to them here:..."
8. **Why the System Failed**: Semantic overlap between lexical features or boundary ambiguity between adjacent categories.
9. **Hypothesis for Improvement**: Fine-tune contrastive embedding weights with class-specific decision margins.
10. **Proposed Next Experiment**: Add intent-specific negative hard-mining samples during retrieval and classification.

---

## Failure Mode 3: Intent Mismatch (ID: `gold_006`)

1. **Failure Mode**: Intent Mismatch on customer support inquiry
2. **Real Customer Message**: "Does no longer sell the lightning-ethernet adapter?"
3. **Gold Intent**: `order_shipping_and_trade_in`
4. **Predicted Intent**: `ios_software_update_bugs`
5. **Gold Decision**: `AUTO_HANDLE`
6. **Predicted Decision**: `ESCALATE_TO_HUMAN`
7. **Relevant Retrieved Evidence**: "Hi! Great question. Our Online Sales Team can help out with keyboards. Reach them here:..."
8. **Why the System Failed**: Semantic overlap between lexical features or boundary ambiguity between adjacent categories.
9. **Hypothesis for Improvement**: Fine-tune contrastive embedding weights with class-specific decision margins.
10. **Proposed Next Experiment**: Add intent-specific negative hard-mining samples during retrieval and classification.

---

## Failure Mode 4: Escalation Mismatch (ID: `gold_007`)

1. **Failure Mode**: Escalation Mismatch on customer support inquiry
2. **Real Customer Message**: "I have the new iPhone 8+ and sometimes when accessing the camera from the lock screen, I get a black screen with the spinner."
3. **Gold Intent**: `hardware_audio_and_screen`
4. **Predicted Intent**: `hardware_audio_and_screen`
5. **Gold Decision**: `AUTO_HANDLE`
6. **Predicted Decision**: `ESCALATE_TO_HUMAN`
7. **Relevant Retrieved Evidence**: "We'd be happy to do everything we can to help. Can you DM us if you have a backup from before your photos went missing?..."
8. **Why the System Failed**: Semantic overlap between lexical features or boundary ambiguity between adjacent categories.
9. **Hypothesis for Improvement**: Fine-tune contrastive embedding weights with class-specific decision margins.
10. **Proposed Next Experiment**: Add intent-specific negative hard-mining samples during retrieval and classification.

---

## Failure Mode 5: Intent Mismatch (ID: `gold_008`)

1. **Failure Mode**: Intent Mismatch on customer support inquiry
2. **Real Customer Message**: "why the hell is the system taking 10 of my 16 gb?????"
3. **Gold Intent**: `general_complaint_and_escalation`
4. **Predicted Intent**: `ios_software_update_bugs`
5. **Gold Decision**: `ESCALATE_TO_HUMAN`
6. **Predicted Decision**: `ESCALATE_TO_HUMAN`
7. **Relevant Retrieved Evidence**: "We can help reclaim that space on your iPhone. Let's get together in DM to take a closer look at what's happening...."
8. **Why the System Failed**: Semantic overlap between lexical features or boundary ambiguity between adjacent categories.
9. **Hypothesis for Improvement**: Fine-tune contrastive embedding weights with class-specific decision margins.
10. **Proposed Next Experiment**: Add intent-specific negative hard-mining samples during retrieval and classification.

---
