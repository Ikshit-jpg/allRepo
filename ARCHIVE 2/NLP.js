// ============================================
// ARCHIVE MINI NLP ENGINE
// ============================================

// ===============================
// MASTER DICTIONARY
// ===============================

const DICTIONARY = [
    "slow", "motion", "fast", "quick",
    "dramatic", "cinematic", "emotional",
    "sad", "dark", "intense",
    "hype", "epic", "energetic",
    "fight", "fights", "dialogue",
    "zoom", "flash", "shake", "glow",
    "music", "soundtrack", "background",
    "very", "extremely"
];

// ===============================
// MAIN ENTRY FUNCTION
// ===============================

function runNLP(inputText) {

    const cleaned = normalize(inputText);
    const tokens = tokenize(cleaned);

    const instructions = {
        mood: detectMood(tokens),
        speed: detectSpeed(tokens),
        effects: detectEffects(tokens),
        focus: detectFocus(tokens),
        music: detectMusic(tokens)
    };

    instructions.confidence = calculateConfidence(instructions);

    return instructions;
}

// ===============================
// NORMALIZATION
// ===============================

function normalize(text) {
    return text.toLowerCase().trim();
}

// ===============================
// TOKENIZATION + FUZZY CORRECTION
// ===============================

function tokenize(text) {
    const words = text.split(/\s+/);
    return words.map(word => fuzzyCorrect(word, DICTIONARY));
}

// ===============================
// LEVENSHTEIN DISTANCE
// ===============================

function levenshtein(a, b) {
    const matrix = [];

    for (let i = 0; i <= b.length; i++) {
        matrix[i] = [i];
    }

    for (let j = 0; j <= a.length; j++) {
        matrix[0][j] = j;
    }

    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            if (b.charAt(i - 1) === a.charAt(j - 1)) {
                matrix[i][j] = matrix[i - 1][j - 1];
            } else {
                matrix[i][j] = Math.min(
                    matrix[i - 1][j - 1] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j] + 1
                );
            }
        }
    }

    return matrix[b.length][a.length];
}

function fuzzyCorrect(word, dictionary) {

    let bestMatch = word;
    let smallestDistance = 3; // tolerance

    dictionary.forEach(dictWord => {
        const distance = levenshtein(word, dictWord);
        if (distance < smallestDistance) {
            smallestDistance = distance;
            bestMatch = dictWord;
        }
    });

    return bestMatch;
}

// ===============================
// MOOD DETECTION (WEIGHTED)
// ===============================

function detectMood(tokens) {

    const moodScores = {
        sad: 0,
        dramatic: 0,
        hype: 0,
        dark: 0
    };

    tokens.forEach((word, index) => {

        const prev = tokens[index - 1];

        const multiplier =
            prev === "very" ? 2 :
            prev === "extremely" ? 3 :
            1;

        if (["sad", "emotional"].includes(word))
            moodScores.sad += multiplier;

        if (["dramatic", "cinematic"].includes(word))
            moodScores.dramatic += multiplier;

        if (["hype", "epic", "energetic"].includes(word))
            moodScores.hype += multiplier;

        if (["dark", "intense"].includes(word))
            moodScores.dark += multiplier;
    });

    return getHighestScore(moodScores);
}

// ===============================
// SPEED DETECTION
// ===============================

function detectSpeed(tokens) {

    let slow = 0;
    let fast = 0;

    tokens.forEach(word => {
        if (word === "slow" || word === "motion")
            slow++;

        if (word === "fast" || word === "quick")
            fast++;
    });

    if (slow > fast) return "slow motion";
    if (fast > slow) return "fast cuts";

    return null;
}

// ===============================
// EFFECTS DETECTION
// ===============================

function detectEffects(tokens) {

    const effects = [];

    tokens.forEach(word => {
        if (word === "zoom") effects.push("zoom effect");
        if (word === "flash") effects.push("flash transition");
        if (word === "shake") effects.push("shake effect");
        if (word === "glow") effects.push("glow effect");
    });

    return effects;
}

// ===============================
// FOCUS DETECTION
// ===============================

function detectFocus(tokens) {

    if (tokens.includes("fight") || tokens.includes("fights"))
        return "fight scenes";

    if (tokens.includes("dialogue"))
        return "dialogue scenes";

    return null;
}

// ===============================
// MUSIC DETECTION
// ===============================

function detectMusic(tokens) {

    if (tokens.includes("sad") && tokens.includes("music"))
        return "sad soundtrack";

    if (tokens.includes("epic") && tokens.includes("music"))
        return "epic soundtrack";

    if (tokens.includes("music") || tokens.includes("soundtrack"))
        return "background music";

    return null;
}

// ===============================
// CONFIDENCE CALCULATION
// ===============================

function calculateConfidence(obj) {

    let filled = 0;

    Object.keys(obj).forEach(key => {
        if (key !== "confidence") {
            const value = obj[key];
            if (value && (Array.isArray(value) ? value.length > 0 : true)) {
                filled++;
            }
        }
    });

    return Math.round((filled / 5) * 100);
}

// ===============================
// SCORE HELPER
// ===============================

function getHighestScore(scores) {

    let highest = 0;
    let winner = null;

    Object.keys(scores).forEach(key => {
        if (scores[key] > highest) {
            highest = scores[key];
            winner = key;
        }
    });

    return winner;
}
