# BGI Trident vs DoorDash GNN: Prior Art Comparison

## DoorDash Approach (Published June 2024)

DoorDash uses GNNs to personalize restaurant recommendation notifications.

### Their Architecture

| Component        | DoorDash Implementation                              |
|------------------|-------------------------------------------------------|
| Graph Type       | Heterogeneous (users + restaurants)                   |
| Data Window      | 1 year of user-restaurant interactions                |
| GNN Model        | ID-GNN (Identity-aware GNN)                           |
| Task             | Link prediction: P(user orders from restaurant in 7d) |
| Node Features    | User attributes + restaurant attributes               |
| Edge Types       | Single: order interactions                            |
| Output           | User and restaurant embeddings                        |
| Serving          | Batch/offline (notifications sent once per day)       |
| Post-Processing  | Geolocation filter, freshness, rating, deduplication  |
| End Action       | Push notification recommending a restaurant           |

### What DoorDash Gets Right

- Heterogeneous graph captures richer relationships than collaborative filtering
- ID-GNN handles structural identity, outperforming vanilla GCN/GAT
- Geolocation-aware candidate selection (geohash + distance filtering)
- Production-proven at DoorDash scale

### Where DoorDash Stops

1. **Single domain**: Food delivery only. No grocery, no dine-in.
2. **Passive output**: Generates a notification. User must still open app, browse, build cart, order.
3. **No execution loop**: The GNN predicts; nothing acts on the prediction autonomously.
4. **No cross-domain signals**: Cannot learn that food orders correlate with grocery purchases or dining bookings.
5. **Batch only**: Daily notification cadence. No real-time session-level intelligence.

## BGI Trident Approach

### Architecture Differences

| Component        | BGI Trident Implementation                            |
|------------------|-------------------------------------------------------|
| Graph Type       | Heterogeneous (users + restaurants + products + venues + time + location) |
| Data Window      | Rolling (continuous update from completed transactions)|
| GNN Models       | Prong 1: PyG ID-GNN (structural) + Prong 2: DGL R-GCN (temporal) |
| Tabular Model    | Prong 3: XGBoost on engineered consumption features   |
| Ensemble         | Stacked generalization meta-learner + Platt calibration|
| Task             | Multi-task: food link prediction + grocery reorder prediction + dineout affinity |
| Node Features    | 6 node types with domain-specific feature vectors     |
| Edge Types       | 8 cross-domain edge types (see graph-schema.md)       |
| Output           | Ranked action plan (what to order, from where, when)  |
| Serving          | Real-time (session-level) + batch (proactive triggers) |
| Post-Processing  | Constraint resolution (open? in stock? slot available?)|
| End Action       | Autonomous execution via MCP APIs with confirmation   |

### Key Architectural Advances

**1. Three-Prong Ensemble (vs DoorDash's Single Model)**

DoorDash uses a single ID-GNN. Trident uses three complementary models:
- PyG ID-GNN captures structural topology (who orders what from where)
- DGL R-GCN captures temporal dynamics (recency decay, time-of-day patterns)
- XGBoost captures tabular signals GNNs miss (sharp thresholds, multiplicative feature interactions)

A user who ordered biryani every Thursday for 6 weeks but stopped 3 weeks ago: PyG still shows strong structural affinity (many edges). DGL catches the drift (decaying edge weights). XGBoost flags the recency gap directly. The ensemble resolves the conflict -- no single model can.

**2. Cross-Domain Edges**

The `OFTEN_PAIRED` edge (Restaurant to Product) and `FOLLOWED_BY_DINING` edge (Restaurant to Venue) enable predictions that span domain boundaries. Example: the graph learns that users who order from Meghana's on weekday evenings frequently purchase Coke from Instamart in the same session, and book similar Andhra restaurants on Dineout within 48 hours.

DoorDash has no equivalent. Their graph cannot surface "you ordered biryani, you probably need Coke" because grocery products are not nodes in their graph.

**3. Temporal Modeling (DGL Prong)**

DoorDash's link prediction is a 7-day probability with no temporal modeling. Trident's DGL prong (Prong 2) uses R-GCN with recency-decayed edge weights and time-of-day/day-of-week features. The system knows this user orders lunch at 12:30 PM on weekdays from a different set of restaurants than dinner at 8 PM. Time is a first-class node type, not just a feature.

**4. Predict-Decide-Execute (Closed Loop)**

DoorDash: GNN predicts -> notification sent -> user acts (or ignores).
Trident: GNN predicts -> orchestrator decomposes into agent tasks -> agents check constraints against live MCP APIs -> confirmation gate -> transaction executes -> graph updates with new edges.

The closed loop means every interaction improves the next prediction. DoorDash's batch system updates daily. Trident's graph updates per-transaction.

**4. Multi-Agent Orchestration**

A single user utterance like "order my usual and get groceries" triggers parallel execution across Food Agent and Instamart Agent, with shared state (delivery address, payment method) managed by the Trident coordinator. DoorDash's system has no multi-agent concept because it operates in a single domain.

## Summary

| Dimension           | DoorDash           | BGI Trident              |
|---------------------|--------------------|--------------------------|
| Domains             | 1 (Food)           | 3 (Food + Instamart + Dineout) |
| Model               | Single ID-GNN      | Three-prong ensemble (PyG + DGL + XGBoost) |
| Graph Nodes         | 2 types            | 6 types                  |
| Graph Edges         | 1 type             | 8 types (cross-domain)   |
| Temporal Modeling    | None (static graph) | DGL prong with recency decay weights |
| Tabular Features    | None               | XGBoost prong (frequency, AOV, basket patterns) |
| Output              | Notification       | Executed transaction      |
| Loop                | Open (predict only) | Closed (predict-decide-execute-learn) |
| Serving             | Batch (daily)      | Real-time + batch         |
| User Action Needed  | Full ordering flow  | Confirmation only         |
| Data Moat           | Order history       | Cross-domain consumption graph |
