"""English hallucination detection template from the paper appendix."""

import re

HALLUCINATION_DETECTION_PROMPT = """# Role
You are a strict auditor of response accuracy. Your core competency lies in meticulously cross-referencing statements within a response text against provided reference materials. Your objective is to verify whether every informational statement in the response text aligns with the reference materials, and to precisely pinpoint any specific erroneous segments.

# Task
I will provide a "Response Text" and a set of "Reference Materials." Your task is to verify, sentence by sentence, every informational statement within the Response Text -- specifically validating whether its assertions are consistent with the content found in the Reference Materials -- and to precisely identify any specific erroneous segments.

# Input Data
## [User Query & Reference Materials]
{{user_prompt}}

## [Response Text]
{{response}}

# Workflow
1. **Extraction:** Extract all informational sentences from the "Response Text." 
- Extracted sentences typically contain specific data points -- such as figures (dates, times, and various other numerical values), entities (people, places, venues, etc.), and logical relationships (e.g., whether a cause-and-effect link holds true).
2. **Localization:** Locate the corresponding paragraphs within the "Reference Materials."
3. **Comparison:** Compare the statements made in the response against the original wording found in the reference materials.
4. **Verdict:**
- **Correct:** The statement aligns perfectly with the content found in the reference materials. 
- **Incorrect:** A corresponding statement exists in the reference materials, but the response's phrasing does not match it; *or* the statement cannot be found within the specified reference materials (i.e., it is fabricated/hallucinated).
5. **Error Localization:** For every incorrect item, further pinpoint the specific "erroneous segment" (`error_spans`) within the "Response Text" itself, to facilitate subsequent token-level penalties. 
- For each incorrect item, you must identify the minimal erroneous segment within the [Response Text] and output it to the `error_spans` list. 
- Each entry in `error_spans` must be a contiguous substring taken directly from the original Response Text. 
- If a single error spans multiple non-contiguous segments, split them into separate elements and add each to the `error_spans` list. 
- Avoid marking entire sentences as erroneous segments; instead, identify only the minimal, specific error fragments. 
- If no errors are detected, the `error_spans` list must be empty. # Possible Errors All possible data accuracy error types:

- conflict: Factual contradiction (the model generates information that contradicts facts in the references, such as misattribution of dates, metrics, entities, etc.)

- fabrication: Fabrication (the model generates information that is not mentioned in the original text/context)

- no_error: No error

# Output Format Please analyze carefully. The output should be in the following format. Strictly adhere to the format and do not output any extra symbols:

### Information Accuracy Analysis

{
"details": [

{
"claim_text": "The extracted sentence containing the information",

"source_text": "The corresponding sentence in the references",

"analysis": "Analyze whether there is a match or whether the calculation result is correct",

"error_type": "One of the error types in Possible Errors: conflict/fabrication/no_error",

"judgment_result": "Correct/Incorrect",

"error_spans": ["Error Span 1", "Error Span 2", ...]

},

{ "claim_text": "The extracted sentence containing the information",

"source_text": "The corresponding sentence from the references",

"analysis": "Analysis of whether a match is found or whether the calculation result is correct",

"error_type": "One of the error types in Possible Errors: conflict/fabrication/no_error",

"judgment_result": "Correct/Incorrect",

"error_spans": ["Error Span 1", "Error Span 2", ...]

},...

]
}
### Summary of Response Accuracy The response contains X claims, of which Y are correct matches and Z are incorrect matches (Note that this summary must strictly adhere to the format; do not output any extra symbols). If illusions exist, summarize the illusion points.
"""

PROMPTS = {task: HALLUCINATION_DETECTION_PROMPT for task in ("financial", "confiqa", "ragtruth")}


def render_prompt(task, query, answer, ground_truth=None):
    """Render the paper template; ground_truth is retained for caller compatibility.

    Claim verification uses the query and reference materials, as in the paper.
    Reference-answer fields remain available for separate rule-based evaluation.
    """
    if task not in PROMPTS:
        raise ValueError(f"Unknown Judge task: {task!r}")
    values = {"user_prompt": query, "response": answer}
    # Replace once so placeholders appearing inside input text stay literal.
    return re.sub(r"\{\{(user_prompt|response)\}\}", lambda match: values[match.group(1)], PROMPTS[task])
