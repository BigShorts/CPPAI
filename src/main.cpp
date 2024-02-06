#include <iostream>
#include <string>

#include "LLM.hpp"

std::string model = "openhermes-2.5-mistral-7b.Q5_K_M.gguf";

int main() {
    LLM llm("models/" + model, 1234, 4096, 1, 1, 33, false);

    if (!llm.ok()) {
        for (auto error : llm.getErrors()) {
            std::cerr << error << "\n";
        }

        return 1;
    }

    while (true) {
        std::cout << "\nEnter a prompt: ";
        std::string prompt;
        std::getline(std::cin, prompt);

        if (prompt == "exit") {
            break;
        }

        std::cout << llm.response(prompt, 2048, true);
        std::fflush(stdout);
        llm.printTimings();
    }
}
