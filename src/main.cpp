#include <iostream>
#include <string>

#include "LLM.hpp"

std::string model = "openhermes-2.5-mistral-7b.Q5_K_M.gguf";

int main() {
    srand(time(0));
    LLM llm("models/" + model, rand(), 32767, 1, 1, "You are going to be used for dietary and gym reasons", 33, false);

    if (!llm.ok()) {
        for (auto error : llm.getErrors()) {
            std::cerr << error << "\n";
        }

        return 1;
    }

    while (true) {
        std::cout << "\n";
        std::string prompt;
        std::getline(std::cin, prompt);

        if (prompt == "exit") {
            break;
        }

        // std::cout << llm.response(prompt, 256);
        llm.response(prompt, 4096, true); // live = true
        std::fflush(stdout);
        // llm.printTimings();
    }
}
