#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

#include <common/common.h>
#include <llama.h>

#include "LLM.hpp"
// #include "FuncLLM.hpp"

std::string model = "Phi-3-mini-4k-instruct-q4.gguf";

std::string systemPrompts[] = {
    "You are going to be used for dietary and gym reasons.",
    "You are a functional AI that is in development and will be used to automate tasks.",
    "You are an AI that is tasked with summarizing news articles. Your input will be a single article and your output explicitly must only be a single short summary."
};

int main() {
    srand(time(0));
    // FuncLLM llm("models/" + model, rand(), 32767, 1, 1, systemPrompts[1], 33, false);
    LLM llm("models/" + model, rand(), 4000, 1, 1, systemPrompts[2], 33, false);

    if (!llm.ok()) {
        for (auto error : llm.getErrors()) {
            std::cerr << error << "\n";
        }

        return 1;
    }


    for (auto entry : std::filesystem::directory_iterator("articles")) {
        std::ifstream article(entry.path());
        std::string prompt;
        std::string line;
        while (std::getline(article, line)) {
            prompt += line + '\n';
        }

        while (prompt.back() == '\n') {
            prompt.pop_back();
        }

        article.close();

        std::string summary = llm.response(prompt, 2000, true); // live = true
        std::cout << '\n';
        std::fflush(stdout);
        llm.printTimings();

        std::ofstream summaryFile("summaries/" + entry.path().filename().string(), std::ios::out);
        summaryFile << summary;
        summaryFile.close();
    }
}
