#include <iostream>
#include <string>
#include <common/common.h>
#include <llama.h>
#include <fstream>

#include "LLM.hpp"
// #include "FuncLLM.hpp"

std::string model = "openhermes-2.5-mistral-7b.Q8_0.gguf";

std::string systemPrompts[] = {
    "You are going to be used for dietary and gym reasons.",
    "You are a functional AI that is in development and will be used to automate tasks.",
    "You are an AI that is tasked with summarizing news articles for the purposes of being read to make informed decisions on the stock market. Your input will be a single article and your output must be only a summary."
};

int main() {
    srand(time(0));
    // FuncLLM llm("models/" + model, rand(), 32767, 1, 1, systemPrompts[1], 33, false);
    LLM llm("models/" + model, rand(), 32767, 1, 1, systemPrompts[2], 33, false);

    if (!llm.ok()) {
        for (auto error : llm.getErrors()) {
            std::cerr << error << "\n";
        }

        return 1;
    }

    std::fstream article("article.txt");

    while (true) {
        std::cout << "Enter prompt:\n";
        std::string prompt;
        std::getline(std::cin, prompt);
        std::cout << '\n';

        if (prompt == "exit") {
            break;
        }

        // std::cout << llm.response(prompt, 256);
        llm.response(prompt, 256, true); // live = true
        std::cout << '\n';
        std::fflush(stdout);
        // llm.printTimings();
    }
}
