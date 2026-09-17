#include "../src/unicode.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

int main() {
    const std::vector<std::string> regex_exprs = {
        "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
    };
    const std::vector<std::string> expected = { " ~", "foo" };
    const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "unexpected split:");
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        return 1;
    }

    const std::vector<std::string> k2_regex = {
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    const std::vector<std::pair<std::string, std::vector<std::string>>> k2_cases = {
        { u8"Hello world", { u8"Hello", u8" world" } },
        { u8"I'm here. We've arrived; they're ready.", { u8"I", u8"'m", u8" here", u8".", u8" We", u8"'ve", u8" arrived", u8";", u8" they", u8"'re", u8" ready", u8"." } },
        { u8"camelCase ABCdef XMLParser", { u8"camelCase", u8" ABCdef", u8" XMLParser" } },
        { u8"1234567 \u0661\u0662\u0663\u0664\u0665\u0666\u0667", { u8"123", u8"456", u8"7", u8" ", u8"\u0661\u0662\u0663", u8"\u0664\u0665\u0666", u8"\u0667" } },
        { u8"  Hello\n\n world\t ", { u8" ", u8" Hello", u8"\n\n", u8" world", u8"\t " } },
        { u8"\u0627\u0644\u0639\u064e\u0631\u064e\u0628\u0650\u064a\u064e\u0651\u0629", { u8"\u0627\u0644\u0639\u064e\u0631\u064e\u0628\u0650\u064a\u064e\u0651\u0629" } },
        { u8"x\u0301y", { u8"x\u0301y" } },
        { u8"a\u200cb", { u8"a\u200cb" } },
        { u8"a\u200db", { u8"a\u200db" } },
        { u8"\u0645\u06cc\u200c\u0631\u0648\u0645", { u8"\u0645\u06cc\u200c\u0631\u0648\u0645" } },
        { u8"\u0915\u094d\u200d\u0937", { u8"\u0915\u094d\u200d\u0937" } },
        { u8"\u200c\u200d", { u8"\u200c\u200d" } },
        { u8"\U0001f469\u200d\U0001f4bb", { u8"\U0001f469\u200d", u8"\U0001f4bb" } },
        { u8"a\u200eb", { u8"a", u8"\u200eb" } },
        { u8"a\u200fb", { u8"a", u8"\u200fb" } },
        { u8"a\u00adb", { u8"a", u8"\u00adb" } },
        { u8"a\ufeffb", { u8"a", u8"\ufeffb" } },
        { u8"a\fb", { u8"a", u8"\fb" } },
        { u8"a\rb", { u8"a", u8"\r", u8"b" } },
        { u8"", {  } },
        { u8"'\u017fa", { u8"'\u017f", u8"a" } },
        { u8"'Sx 'Tx 'REx 'VEx 'Mx 'LLx 'Dx", { u8"'S", u8"x", u8" '", u8"Tx", u8" '", u8"REx", u8" '", u8"VEx", u8" '", u8"Mx", u8" '", u8"LLx", u8" '", u8"Dx" } },
    };
    for (const auto & test : k2_cases) {
        if (unicode_regex_split(test.first, k2_regex, false) != test.second) {
            fprintf(stderr, "unexpected K2 Horizon split for: %s\n", test.first.c_str());
            return 1;
        }
    }

    return 0;
}
