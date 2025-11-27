# Corrected LangGraph Flowchart & System Behavior

This document includes the corrected flow that fully matches the
required LangGraph Agent Behavior and system goals.

------------------------------------------------------------------------

## Corrected Flowchart

``` mermaid
flowchart TD

%% ========== MAIN ENTRY ==========
A[Human Input] --> R{Router}

%% ========== ROUTING LOGIC ==========
R -->|first run / no company profile| W[Welcome / auto greeting]
W --> CPGEN

R -->|company profile missing| CPGEN[Company Profile Generation]

R -->|company profile ready, not confirmed| CPCONF[Company Profile Confirmation]

R -->|company profile confirmed AND seeds < 5| SEEDS[Collect Best Customer Websites]

R -->|company profile confirmed AND enough seeds AND ICP profile missing| ICPGEN[ICP Profile Generation]

R -->|ICP profile ready, not confirmed| ICPCONF[ICP Profile Confirmation]

R -->|both profiles confirmed, ICP discovery pending| ICDPROMPT[ICP Discovery Prompt]

R -->|user accepts micro-ICP| ACCEPT[Accept ICP]

R -->|user initiates explicit discovery command| DISCOVER[Confirm Run]

R -->|has candidates| CAND[Candidate Generation]

R -->|needs enrichment| ENRICH[Enrichment]

R -->|after enrichment| SCORE[Scoring]

R -->|general questions| QA[LeadGen Q&A]


%% ========== COMPANY PROFILE WORKFLOW ==========
CPGEN -->|profile generated| CPCONF
CPCONF -->|user approves| SEEDS
CPCONF -->|user requests edits| CPGEN


%% ========== SEEDS WORKFLOW ==========
SEEDS -->|user provides seeds| ICPGEN
SEEDS -->|not enough seeds| SEEDS


%% ========== ICP PROFILE WORKFLOW ==========
ICPGEN -->|ICP profile generated| ICPCONF
ICPCONF -->|user approves| ICDPROMPT
ICPCONF -->|user requests edits| ICPGEN


%% ========== ICP DISCOVERY ==========
ICDPROMPT -->|user approves discovery| DISCOVER
ICDPROMPT -->|user declines| QA


%% ========== DISCOVERY PIPELINE ==========
DISCOVER --> CAND --> ENRICH --> SCORE

ACCEPT --> ENRICH
```

------------------------------------------------------------------------

## Summary

This flow ensures:

-   Strict separation between Company Profile and ICP Profile workflows\
-   Required explicit confirmation for both profiles\
-   Update/edit loops for both profiles\
-   Deterministic routing with zero mixing of paths\
-   Proper gating before ICP discovery\
-   A complete discovery pipeline after confirmation


## 🔍 Detailed Explanation of Each Step

Below is a precise description of each node and its responsibility.

1. Human Input (A)

Entry point for all user interactions — questions, commands, confirmations, URLs, or edits.

2. Router (R)

The intelligent routing node evaluates system state:

Which profiles exist?

Which profiles are confirmed?

Do we have enough seeds?

Is ICP discovery allowed?

Is the user asking for general info?

This ensures correct, deterministic paths and prevents mixing workflows.

### COMPANY PROFILE WORKFLOW
3. Welcome (W)

Triggered only on:

first run

no Company Profile available

It introduces the system and asks for the user's business website.

4. Company Profile Generation (CPGEN)

The system performs:

website crawling

content extraction

business analysis

Company Profile synthesis

Output is stored as a draft.

5. Company Profile Confirmation (CPCONF)

User is shown the generated Company Profile.

User may:

Approve → proceed

Request edits → regenerate

This loop continues until confirmation.

### BEST CUSTOMER SEEDS WORKFLOW
6. Collect Best Customer Websites (SEEDS)

Only allowed after Company Profile is confirmed.

User must supply ≥5 customer websites.
If fewer than 5 are provided:

The system loops and asks for more.

### ICP PROFILE WORKFLOW
7. ICP Profile Generation (ICPGEN)

Once enough seeds are collected:

Crawl customer websites

Analyze patterns

Extract ICP traits

Synthesizes ICP Profile

Result stored as draft.

8. ICP Profile Confirmation (ICPCONF)

User reviews the ICP Profile.

User may:

Approve → continue

Request edits → regenerate ICP Profile

This loop ensures profile accuracy.

### ICP DISCOVERY
9. ICP Discovery Prompt (ICDPROMPT)

Both Company + ICP Profiles are confirmed.

System asks:
“Would you like to run full ICP discovery?”

User may:

Approve → discovery pipeline

Decline → exit to Q&A

10. Accept Micro-ICP (ACCEPT)

Alternate path where user directly accepts a micro-ICP without full discovery.

Proceeds to enrichment.

### DISCOVERY PIPELINE
11. Confirm Run (DISCOVER)

User explicitly starts ICP Discovery.

12. Candidate Generation (CAND)

System generates potential ICP company candidates based on:

pattern matching

similarity thresholds

industry clustering

13. Enrichment (ENRICH)

Each candidate is enriched with:

firmographics

technographics

revenue data

contact data (if applicable)

14. Scoring (SCORE)

Enriched candidates are scored against the ICP.

Output:

ranked candidate list

scoring rationale

next-step recommendations

15. General Q&A (QA)

Fallback handler for user questions unrelated to workflows.

## 🎉 Summary

This corrected workflow ensures:

Perfect separation of Company vs ICP tasks

Mandatory confirmations before progression

Proper looping for edits

No mixing or skipping of stages

Deterministic routing based on system state

Fully compliant ICP discovery pipeline
