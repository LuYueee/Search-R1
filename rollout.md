# Grounded-R1 Flowchart

This diagram illustrates the four-phase rollout generation process of the Grounded-R1 framework.

```mermaid
flowchart TD
    Start[Input Question] --> LoopStart[Thinking Loop Initiation]
    
    LoopStart --> Think[Generate Reasoning Sentence within <br/>&lt;think&gt;...&lt;/think&gt;]
    Think --> ComputeRHRD
    
    subgraph Step1_Sense [1. Real-Time Hallucination Risk Sensing]
        ComputeRHRD["Compute Sentence-Level Hallucination Risk<br/>RHRD<sub>sent</sub>(t)"]
    end
    
    ComputeRHRD --> PolicyDecision{Policy Network:<br/>Ternary Action Selection}
    
    subgraph Step2_Adapt [2. Adaptive Risk-Aware Decision]
        PolicyDecision
    end

    PolicyDecision -->|Low Hallucination Risk<br/>ACTION: CONTINUE_THINK| LoopStart
    
    PolicyDecision -->|High Hallucination Risk<br/>ACTION: SEARCH| GenQuery["Formulate Retrieval Query within <br/>&lt;search&gt;...&lt;/search&gt;"]
    
    subgraph Step3_Fetch [3. Evidence Retrieval and Buffer Augmentation]
        GenQuery --> Retrieve
        
        subgraph ExternalKnowledge [External Knowledge System]
            Retrieve[Retrieval Engine]
            Retrieve --> KB["Knowledge Base<br/>G<sub>world</sub>"]
        end
        
        KB --> RetrievedDocs[Acquire Relevant Passages]
        RetrievedDocs --> UpdateBuffer["Augment Evidence Buffer<br/>G<sub>src</sub>(t)"]
        UpdateBuffer --> AppendToContext[Inject Evidence into Context within <br/>&lt;information&gt;...&lt;/information&gt;]
    end
    
    AppendToContext --> LoopStart
    
    PolicyDecision -->|Low Hallucination Risk and Sufficient Evidence<br/>ACTION: ANSWER| EndThink[Terminate &lt;think&gt; Segment]

    subgraph Step4_Emit [4. Grounded Content Generation]
        EndThink --> GenAnswer[Generate Final Answer within <br/>&lt;answer&gt;...&lt;/answer&gt;]
        GenAnswer --> EndGenAnswer[Terminate &lt;answer&gt; Segment]
        EndGenAnswer --> Output[Output Grounded Response]
    end

    style Start fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Output fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Step1_Sense fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Step2_Adapt fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Step3_Fetch fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Step4_Emit fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style ExternalKnowledge fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style KB fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,shape:cylinder
    style PolicyDecision fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

## Diagram Description

The flowchart depicts the following four-phase process:

1. **Real-Time Hallucination Risk Sensing**: Computation of sentence-level hallucination risk using RHRD.
2. **Adaptive Risk-Aware Decision**: Policy network selects from three actions based on risk assessment.
3. **Evidence Retrieval and Buffer Augmentation**: External knowledge retrieval and evidence buffer updates.
4. **Grounded Content Generation**: Final answer generation with evidence grounding.
