# Evaluation Report: Final Refinement (Thresh=0.4)

## Overview
- **Total Reports**: 173
- **Strategy**: Acronyms (incl. TCAS/RA) + UUIDs + Thresh 0.4
- **Reclassified "OTHER"**: 117 (67.6%)
- **Unclassified (UNK)**: 56 (Reduced from ~112)
- **MAC Detected**: 18

## Comparison by Method
| Method | Count | Avg Confidence |
|---|---|---|
| Rule (Hybrid) | 0 | 1.00 |
| Embedding | 117 | 0.4653 |
| Low Confidence | 56 | 0.3412 |

## Top High-Confidence Reclassifications

### Row 130 (Embedding)
- **Event**: `Phase: traffic pattern. Actor: ATC, Instructor. Trigger: Near Mid Air Collision event. Outcome: Near Mid Air Collision event.`
- **Predicted**: **MAC** (Airprox, ACAS alerts, loss of separation, near collisions between aircraft)
- **Confidence**: 0.5555

### Row 172 (Embedding)
- **Event**: `Phase: mid air. Actor: ATC, Pilot. Trigger: near mid air collision with another aircraft. Outcome: Collision.`
- **Predicted**: **MAC** (Airprox, ACAS alerts, loss of separation, near collisions between aircraft)
- **Confidence**: 0.5530

### Row 128 (Embedding)
- **Event**: `Phase: landing. Actor: Instructor. Trigger: Near Mid Air Collision event. Outcome: avoided a collision.`
- **Predicted**: **CFIT** (Inflight collision or near collision with terrain, water, or obstacle without indication of loss of control)
- **Confidence**: 0.5521

### Row 49 (Embedding)
- **Event**: `Phase: traffic pattern. Actor: Instructor. Trigger: near mid air collision with departing aircraft. Outcome: near mid air collision.`
- **Predicted**: **MAC** (Airprox, ACAS alerts, loss of separation, near collisions between aircraft)
- **Confidence**: 0.5445

### Row 36 (Embedding)
- **Event**: `Phase: traffic pattern. Actor: Pilot. Trigger: Near Mid Air Collision in the traffic pattern. Outcome: Near Mid Air Collision.`
- **Predicted**: **MAC** (Airprox, ACAS alerts, loss of separation, near collisions between aircraft)
- **Confidence**: 0.5351

### Row 58 (Embedding)
- **Event**: `Phase: landing. Actor: ATC, Pilot, Captain. Trigger: low altitude warning from ATC during an unstable approach in visual conditions. Outcome: returned for a landing.`
- **Predicted**: **UIMC** (Unintended flight in IMC)
- **Confidence**: 0.5344

### Row 79 (Embedding)
- **Event**: `Actor: Controller. Trigger: Near Mid Air Collision with another aircraft. Outcome: Near Mid Air Collision with another aircraft.`
- **Predicted**: **CFIT** (Inflight collision or near collision with terrain, water, or obstacle without indication of loss of control)
- **Confidence**: 0.5339

### Row 144 (Embedding)
- **Event**: `Phase: cruise. Actor: Pilot. Trigger: Controlled Flight Into Terrain event. System: aircraft control. Outcome: temporary loss of aircraft control and a Controlled Flight Into Terrain event.`
- **Predicted**: **CFIT** (Inflight collision or near collision with terrain, water, or obstacle without indication of loss of control)
- **Confidence**: 0.5278

### Row 93 (Embedding)
- **Event**: `Phase: landed. Actor: Pilot. Trigger: Near Mid Air Collision. Outcome: Near Mid Air Collision.`
- **Predicted**: **CFIT** (Inflight collision or near collision with terrain, water, or obstacle without indication of loss of control)
- **Confidence**: 0.5274

### Row 28 (Embedding)
- **Event**: `Phase: approach. Actor: Controller, Pilot, Tower. Trigger: Controlled Flight Into Terrain event. System: lost situational awareness. Outcome: Controlled Flight Into Terrain event.`
- **Predicted**: **CFIT** (Inflight collision or near collision with terrain, water, or obstacle without indication of loss of control)
- **Confidence**: 0.5262
