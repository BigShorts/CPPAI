#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

#include <common/common.h>
#include <llama.h>

#include "LLM.hpp"

std::string model = "Phi-3-mini-4k-instruct-q4.gguf";
//std::string model = "openhermes-2.5-mistral-7b.Q8_0.gguf";

std::string systemPrompts[] = {
    "You are going to be used for dietary and gym reasons.",
    "You are a functional AI that is in development and will be used to automate tasks.",
    "You are a system prompt generation ai. Only generate a prompt and nothing more.",
    "You are an LLM tasked with summarizing news articles in a concise manner. The generated output should contain only the key points or highlights of the article, and absolutely no additional information other than what was mentioned in the article. The objective is to allow the reader to get the main essence of the article in a brief and impactful manner without any unnecessary details. Make sure to keep the summary informative, relevant, and easy to understand.You are an LLM tasked with summarizing news articles in a concise manner. The generated output should contain only the key points or highlights of the article, and absolutely no additional information other than what was mentioned in the article. The objective is to allow the reader to get the main essence of the article in a brief and impactful manner without any unnecessary details. Make sure to keep the summary informative, relevant, and easy to understand. Write a single paragraph. Do not start a new response."
};

int main() {
    srand(time(0));
    // FuncLLM llm("models/" + model, rand(), 32767, 1, 1, systemPrompts[1], 33, false);
    LLM llm("models/" + model, rand(), 4000, 1, 1, systemPrompts[3], 33, false);

    if (!llm.ok()) {
        for (auto error : llm.getErrors()) {
            std::cerr << error << "\n";
        }

        return 1;
    }

    // while (true) {
    //     std::cout << llm.response("Generate a prompt for an LLM tasked with summarising news articles. Prevent the LLM from generating absolutely anything but the summary, this is very important so stress this.", 2000, true) << '\n';
    //     llm.printTimings();
    // }

    while (true)
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

        std::string summary = llm.response(prompt, 400, true); // live = true
        std::cout << '\n';
        std::fflush(stdout);
        llm.printTimings();

        std::ofstream summaryFile("summaries/" + entry.path().filename().string(), std::ios::out);
        summaryFile << summary;
        summaryFile.close();
    }
}
