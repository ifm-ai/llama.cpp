#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <vector>

struct regex_split_case {
    const char * description;
    std::vector<std::string> regex_exprs;
    std::string input;
    std::vector<std::string> expected;
};

static bool run_case(const regex_split_case & test_case) {
    const auto actual = unicode_regex_split(test_case.input, test_case.regex_exprs, false);

    if (actual == test_case.expected) {
        return true;
    }

    fprintf(stderr, "FAILED: %s\nunexpected split:", test_case.description);
    for (const auto & piece : actual) {
        fprintf(stderr, " [%s]", piece.c_str());
    }
    fprintf(stderr, "\n");
    return false;
}

int main() {
    // The K2 Horizon / MoVA pre-tokenizer regex (tokenizer.ggml.pre == "k2-horizon").
    // Unlike llama3's near-identical pattern, its word-run group also admits combining
    // marks (\p{M}) and the ZWNJ/ZWJ joiners (‌/‍) used by many Indic scripts,
    // so a plain string-match against unicode_regex_split_custom_llama3 would miss it and
    // fall through to the std::regex fallback, which cannot compile ‌/‍ at all.
    const std::vector<std::string> k2_horizon_regex = {
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };

    const std::vector<regex_split_case> cases = {
        {
            "original custom regex still splits correctly",
            { "[~][A-Za-z]+| ?[\\p{S}]+|\\s+" },
            " ~foo",
            { " ~", "foo" },
        },
        {
            "k2-horizon: leading space attaches to the following word",
            k2_horizon_regex,
            "foo bar",
            { "foo", " bar" },
        },
        {
            "k2-horizon: case-insensitive contraction splits off the suffix",
            k2_horizon_regex,
            "isn't",
            { "isn", "'t" },
        },
        {
            "k2-horizon: a combining mark stays attached to its base letter",
            k2_horizon_regex,
            "é", // "e" + COMBINING ACUTE ACCENT (U+0301)
            { "é" },
        },
        {
            "k2-horizon: a ZWJ between two letters keeps the word joined",
            k2_horizon_regex,
            "a‍b", // "a" + ZERO WIDTH JOINER (U+200D) + "b"
            { "a‍b" },
        },
        {
            "k2-horizon: digit runs are grouped into chunks of at most three",
            k2_horizon_regex,
            "12345",
            { "123", "45" },
        },
    };

    int failures = 0;
    for (const auto & test_case : cases) {
        if (!run_case(test_case)) {
            failures++;
        }
    }

    return failures == 0 ? 0 : 1;
}
