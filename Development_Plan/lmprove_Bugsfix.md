# LangGraph AI Agents System Prompt

## Goal

Ensure the system correctly collects, generates, persists, updates, and
confirms two separate profile types:

1.  **Company Profile** -- extracted from the user's business website.\
2.  **ICP (Ideal Customer Profile)** -- extracted from the best
    customer's website(s).

------------------------------------------------------------------------

## System Requirements

### 1. Profile Collection & Generation

-   The agent must crawl and extract data from:
    -   The user's business website → generate **Company Profile**
    -   Each best customer website provided by the user → generate **ICP
        Profile**
-   The agent must always be aware of which profile it is generating,
    with no mixing.

### 2. Persistence

-   Each profile must be saved with clear identifiers:
    -   `company_profile`
    -   `icp_profile_<customer_index>`

### 3. User Review & Editing

-   The system must:
    -   Present each generated profile to the user
    -   Allow the user to request edits or updates
    -   Regenerate or update the profile upon request
-   A profile is marked **confirmed** only when explicitly confirmed by
    the user.

### 4. Confirmation Logic & Flow Control

-   Routing rules:
    -   If **both** Company Profile **and** ICP Profiles are confirmed →
        proceed to **ICP Discovery Confirmation**
    -   If **any profile is missing or unconfirmed** → return to the
        generation/update step for that profile
-   No cross-contamination between Company Profile and ICP Profiles is
    allowed.

### 5. Routing Rules

-   The state machine must separate:
    -   Company profile workflow
    -   ICP profile workflow
-   Agents must follow the correct paths with no mixing or overwriting.

------------------------------------------------------------------------

## Expected Behavior

-   The system always knows:
    -   Which profile is being worked on
    -   Whether it's generated, updated, or confirmed
    -   Whether it should proceed to ICP discovery or return to missing
        tasks
-   Workflow must be deterministic, bug-free, and perfectly routed.
