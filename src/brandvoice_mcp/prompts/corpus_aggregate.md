You are synthesizing a single voice profile from multiple writing excerpts by the same author.
The excerpts are separated by "---" and may overlap in topic; infer consistent style metrics across all of them.

Return a JSON object with these exact fields:

- avg_sentence_length: average words per sentence across the combined corpus (float)
- vocabulary_richness: ratio of unique words to total words, 0-1 (float)
- formality_score: 0 = very casual, 1 = very formal (float)
- humor: 0 = serious, 1 = playful or humorous (float)
- technical_depth: 0 = accessible / non-technical, 1 = expert jargon and dense detail (float)
- warmth: 0 = detached, 1 = warm, personal, or empathetic (float)
- dominant_tone: one of "conversational", "authoritative", "playful", "academic", "professional", "inspirational" (string)
- rhetorical_patterns: up to 5 recurring patterns, e.g. ["uses analogies", "short punchy sentences"] (list of strings)

CORPUS (excerpt blocks separated by ---):
{corpus}

Return ONLY valid JSON, no markdown fencing.
