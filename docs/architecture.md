# BGI Trident - System Architecture

## High-Level Flow

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        Voice["Voice (Kannada/Hindi/Tamil)"]
        Text["Text/Chat"]
        Proactive["Proactive Trigger (Kumo)"]
    end

    subgraph VPA["Voice Provider Abstraction"]
        VP{{"VoiceProvider Protocol"}}
        Vapi["Vapi Adapter"]
        Bolna["Bolna Adapter"]
        Retell["Retell Adapter"]
        VP --- Vapi
        VP --- Bolna
        VP --- Retell
    end

    subgraph NLU["Swar NLU Engine"]
        Intent["Intent Classifier"]
        Entity["Entity Extractor"]
        CodeMix["Code-Mix Handler"]
    end

    subgraph Kumo["Kumo Graph Engine (Three-Prong Ensemble)"]
        GraphBuilder["Heterogeneous Graph Builder"]
        
        subgraph Prong1["Prong 1: PyG"]
            IDGNN["ID-GNN Encoder"]
            PyGEmbed["Structural Embeddings (128d)"]
            PyGLP["Link Prediction Head"]
        end

        subgraph Prong2["Prong 2: DGL"]
            RGCN["R-GCN Encoder"]
            TemporalDecay["Recency Decay Weights"]
            DGLEmbed["Temporal Embeddings (128d)"]
            DGLLP["Link Prediction Head"]
        end

        subgraph Prong3["Prong 3: XGBoost"]
            FeatEng["Feature Engineering"]
            XGB["Gradient Boosted Trees"]
            XGBScore["Tabular Scores"]
        end

        Ensemble["Ensemble Meta-Learner\n(Stacked Generalization)"]
        GraphUpdate["Online Graph Updater"]

        GraphBuilder --> IDGNN --> PyGEmbed --> PyGLP
        GraphBuilder --> RGCN
        TemporalDecay --> RGCN
        RGCN --> DGLEmbed --> DGLLP
        GraphBuilder --> FeatEng --> XGB --> XGBScore
        PyGLP & DGLLP & XGBScore --> Ensemble
    end

    subgraph Trident["Trident Orchestrator"]
        Coordinator["Multi-Agent Coordinator"]
        StateMgr["Session State Manager"]
        ConfGate["Confirmation Gate"]
        Fallback["Fallback Engine"]
    end

    subgraph Agents["Domain Agents"]
        FoodAgent["Food Agent"]
        InstamartAgent["Instamart Agent"]
        DineoutAgent["Dineout Agent"]
    end

    subgraph MCP["Swiggy MCP Servers"]
        FoodMCP["Food MCP\nsearch | menu | cart | order | track"]
        InstamartMCP["Instamart MCP\nsearch | cart | checkout | track"]
        DineoutMCP["Dineout MCP\nsearch | details | slots | book"]
    end

    Voice --> VP
    VP --> NLU
    Text --> NLU
    Proactive --> Coordinator

    NLU --> Coordinator
    Ensemble --> Coordinator

    Coordinator --> FoodAgent
    Coordinator --> InstamartAgent
    Coordinator --> DineoutAgent
    Coordinator --> StateMgr
    Coordinator --> ConfGate

    FoodAgent --> FoodMCP
    InstamartAgent --> InstamartMCP
    DineoutAgent --> DineoutMCP

    FoodMCP --> GraphUpdate
    InstamartMCP --> GraphUpdate
    DineoutMCP --> GraphUpdate
    GraphUpdate --> GraphBuilder

    ConfGate --> Fallback
```

## Predict-Decide-Execute-Learn Loop

```mermaid
sequenceDiagram
    participant U as User (Voice/Text)
    participant NLU as Swar NLU
    participant K as Kumo (PyG+DGL+XGB)
    participant T as Trident Coordinator
    participant FA as Food Agent
    participant IA as Instamart Agent
    participant DA as Dineout Agent
    participant CG as Confirmation Gate
    participant G as Graph Updater

    U->>NLU: "Meghana's se biryani order karo,<br/>Harpic bhi chahiye,<br/>Saturday ko table book karo"

    NLU->>T: Intent: MULTI_ORDER<br/>Entities: {restaurant: Meghana's,<br/>item: biryani, product: Harpic,<br/>day: Saturday}

    T->>K: Get ensemble predictions for user_42
    K-->>T: Ensemble scores (PyG+DGL+XGB):<br/>Meghana's (0.94), Coke pairing (0.87),<br/>Koramangala venues (0.72)

    par Parallel Agent Execution
        T->>FA: search_restaurants("Meghana's")<br/>+ get_menu + update_cart
        T->>IA: search_products("Harpic")<br/>+ update_cart
        T->>DA: search_restaurants_dineout(cuisine="South Indian")<br/>+ get_available_slots(Saturday)
    end

    FA-->>T: Cart ready: Biryani Rs 350
    IA-->>T: Cart ready: Harpic Rs 189
    DA-->>T: Slot available: Sat 8PM, 4 people

    T->>K: Cross-domain bundle:<br/>suggest Coke pairing? (0.87)
    T->>U: "Biryani from Meghana's Rs 350,<br/>Harpic Rs 189 from Instamart,<br/>table at Farzi Cafe Saturday 8PM.<br/>Also adding Coke? You usually pair it.<br/>Total Rs 579. Confirm?"

    U->>CG: "Haan, Coke bhi add karo"

    CG->>FA: place_food_order()
    CG->>IA: checkout()
    CG->>DA: book_table()

    FA-->>G: Order completed (Meghana's, biryani, Rs 350)
    IA-->>G: Order completed (Harpic + Coke, Rs 229)
    DA-->>G: Booking confirmed (Farzi Cafe, Sat 8PM)

    G->>K: Update edges:<br/>ORDERED_FROM weight++<br/>PURCHASED weight++<br/>BOOKED_AT new edge<br/>OFTEN_PAIRED confirmed
```

## Graph Construction Pipeline

```mermaid
flowchart LR
    subgraph Data["Raw Interaction Data"]
        FO["Food Orders\n50K records"]
        IO["Instamart Orders\n20K records"]
        DB["Dineout Bookings\n5K records"]
    end

    subgraph Build["Graph Builder"]
        NF["Node Feature\nEngineering"]
        EW["Edge Weight\nComputation"]
        CD["Cross-Domain\nEdge Discovery"]
        TF["Temporal Feature\nExtraction"]
    end

    subgraph Graph["Heterogeneous Graph"]
        UN["User Nodes (500)"]
        RN["Restaurant Nodes (200)"]
        PN["Product Nodes (500)"]
        VN["Venue Nodes (100)"]
        TN["TimeSlot Nodes (168)"]
        LN["Location Nodes (50)"]
    end

    subgraph Train["Three-Prong Training"]
        direction TB
        subgraph P1["Prong 1: PyG"]
            ENC["ID-GNN Encoder"]
        end
        subgraph P2["Prong 2: DGL"]
            TGAT["R-GCN + Temporal Decay"]
        end
        subgraph P3["Prong 3: XGBoost"]
            XGB["Gradient Boosted Trees"]
        end
        ENS["Ensemble Meta-Learner"]
    end

    subgraph Output["Predictions"]
        UE["User Embeddings (PyG 128d + DGL 128d)"]
        RE["Restaurant Embeddings (PyG 128d + DGL 128d)"]
        PE["Product Embeddings (PyG 64d + DGL 64d)"]
        VE["Venue Embeddings (PyG 64d + DGL 64d)"]
        XS["XGBoost Tabular Scores"]
        FP["Final Ensemble Probabilities"]
    end

    FO --> NF
    IO --> NF
    DB --> NF
    FO --> EW
    IO --> EW
    DB --> EW
    FO & IO --> CD
    FO & DB --> CD
    FO & IO & DB --> TF

    NF --> UN & RN & PN & VN
    EW --> Graph
    CD --> Graph
    TF --> TN & LN

    Graph --> ENC
    Graph --> TGAT
    Graph --> XGB

    ENC --> UE & RE & PE & VE
    TGAT --> UE & RE & PE & VE
    XGB --> XS
    UE & RE & PE & VE & XS --> ENS --> FP
```

## Confirmation Gate (Fraud/Risk Pattern)

```mermaid
stateDiagram-v2
    [*] --> PendingConfirmation: Agent execution complete

    PendingConfirmation --> ReadBack: Generate summary
    ReadBack --> AmountSpoken: State total in user's language
    AmountSpoken --> WaitingConfirm: Await explicit confirmation

    WaitingConfirm --> Confirmed: "haan" / "yes" / "sari"
    WaitingConfirm --> Modified: "biryani hatao" / "change restaurant"
    WaitingConfirm --> Cancelled: "cancel" / "ruko"
    WaitingConfirm --> TimedOut: 30s no response

    Confirmed --> ExecuteTransaction: Fire MCP transactional calls
    Modified --> RerunAgents: Re-execute with modifications
    Cancelled --> SessionEnd: Log cancellation
    TimedOut --> Retry: One retry prompt

    Retry --> WaitingConfirm: Re-prompt
    Retry --> SessionEnd: Second timeout

    ExecuteTransaction --> GraphUpdate: Update behavioral graph
    ExecuteTransaction --> AuditLog: Log confirmation event
    GraphUpdate --> [*]
```

## Mock-to-Live MCP Swap

```mermaid
flowchart TB
    Config["MCP_MODE env var"]

    Config -->|"mock"| MockFactory["Mock MCP Factory"]
    Config -->|"live"| LiveFactory["Live MCP Factory"]

    MockFactory --> MockFood["MockFoodMCP\n(fixtures/restaurants.json)"]
    MockFactory --> MockInstamart["MockInstamartMCP\n(fixtures/products.json)"]
    MockFactory --> MockDineout["MockDineoutMCP\n(fixtures/venues.json)"]

    LiveFactory --> LiveFood["SwiggyFoodMCP\n(API credentials)"]
    LiveFactory --> LiveInstamart["SwiggyInstamartMCP\n(API credentials)"]
    LiveFactory --> LiveDineout["SwiggyDineoutMCP\n(API credentials)"]

    MockFood & LiveFood --> FoodAgent["Food Agent"]
    MockInstamart & LiveInstamart --> InstamartAgent["Instamart Agent"]
    MockDineout & LiveDineout --> DineoutAgent["Dineout Agent"]

    style Config fill:#f9f,stroke:#333
    style MockFactory fill:#bbf,stroke:#333
    style LiveFactory fill:#bfb,stroke:#333
```
