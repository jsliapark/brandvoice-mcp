Analyze the following writing sample and extract style metrics.
Return a JSON object with these exact fields:

- avg_sentence_length: average number of words per sentence (float)
- vocabulary_richness: ratio of unique words to total words, 0-1 (float)
- formality_score: 0 = very casual, 1 = very formal (float)
- dominant_tone: one of "conversational", "authoritative", "playful", "academic", "professional", "inspirational" (string)
- rhetorical_patterns: list of patterns observed, e.g. ["uses analogies", "starts with questions", "short punchy sentences"] (list of strings, max 5)

WRITING SAMPLE:
{content}

Return ONLY valid JSON, no markdown fencing.
