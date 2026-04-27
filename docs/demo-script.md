# Demo Script: 2-Minute Video Walkthrough

## Target Audience
Swiggy Builders Club engineering reviewers.

## Setup
- Split screen: left = terminal/voice UI, right = live graph visualization
- Graph visualization shows nodes lighting up as the agent processes
- Mock MCP mode (synthetic Bangalore data)

## Script

### 0:00 - 0:15 | Hook
"This is Trident. It does not wait for you to open Swiggy. It already knows what you need."

Show the behavioral graph for user_42 with existing edges: frequent orders from Meghana's on Thursdays, Instamart Coke purchases, weekend Dineout bookings.

### 0:15 - 0:45 | Proactive Trigger
Thursday 7:30 PM. Kumo's GNN fires a prediction: user_42 has a 94% probability of ordering from Meghana's tonight, 87% probability of pairing with Coke, 72% probability of booking a weekend table.

The agent initiates (not the user):
"Ninna usual Thursday biryani Meghana's inda order maaDla? Coke nu beku?" (Kannada)
Translation subtitle: "Shall I order your usual Thursday biryani from Meghana's? Need Coke too?"

### 0:45 - 1:15 | Cross-Domain Execution
User responds in Kannada: "Haan, biryani order maaDu, Harpic nu add maaDu, Saturday ge table book maaDu."
Translation: "Yes, order biryani, add Harpic too, book a table for Saturday."

Show three parallel API calls firing simultaneously:
- Food Agent: search_restaurants -> get_restaurant_menu -> update_food_cart
- Instamart Agent: search_products("Harpic") -> update_cart
- Dineout Agent: search_restaurants_dineout -> get_available_slots

Graph visualization: new edges lighting up in real-time across all three domains.

### 1:15 - 1:35 | Confirmation Gate
Agent reads back in Kannada:
"Meghana's biryani Rs 350, Instamart inda Harpic Rs 189, Saturday 8 PM ge Farzi Cafe nalli table. Total Rs 539. Confirm maaDla?"

User: "Haan."

Three transactional calls fire: place_food_order, checkout, book_table.
Graph updater adds new edges, increments weights.

### 1:35 - 1:50 | Fallback Demo
"What if Meghana's is closed?"

Show the fallback engine: Trident checks the graph for the next-highest-affinity restaurant with similar cuisine. Suggests Nandhana Palace (affinity score 0.81).

"Meghana's close aagide. Nandhana Palace inda similar biryani ide, 25 min. Order maaDla?"

### 1:50 - 2:00 | Close
"Voice is the interface. The graph is the intelligence. Trident predicts, decides, executes, and learns. Every order makes the next one better."

Show the mock-to-live config swap: `MCP_MODE=live`. Same code, real Swiggy APIs.

## Technical Notes for Recording
- Use screen recording with OBS
- Kannada TTS for agent responses (Vapi or pre-recorded)
- Graph visualization: D3.js force-directed layout with animated edges
- Terminal output showing MCP API calls and responses in real-time
- Keep it tight. No filler. Every second shows something working.
