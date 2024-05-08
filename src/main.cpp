#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

#include <common/common.h>
#include <llama.h>

#include "LLM.hpp"
#include "Formats/ChatML.hpp"

std::string modelPhi3 = "Phi-3-mini-4k-instruct-q4.gguf";
std::string modelChatML = "Hermes-2-Pro-Llama-3-8B-Q8_0.gguf";

void articleSummary();
void chat();

int main() {
    srand(time(0));
    
    //articleSummary();
    chat();
}

std::string summaryPrompt = std::string(
    "You are an LLM tasked with summarizing news articles in a concise manner.\n") +
    "The generated output should contain only the key points or highlights of the article, and absolutely no additional information other than what was mentioned in the article.\b" + 
    "The objective is to allow the reader to get the main essence of the article in a brief and impactful manner without any unnecessary details.\n" + 
    "Make sure to keep the summary informative, relevant. Write a single paragraph. Do not start a new response.";

void articleSummary() {
    ChatML llm("models/" + modelChatML, rand(), 32767, 1, 1, LLMType::CHAT, summaryPrompt, 33, false);

    if (!llm.ok()) {
        for (auto error : llm.getErrors()) {
            std::cerr << error << "\n";
        }

        exit(1);
    }

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
        std::cout << "\n--------------\n";
        std::fflush(stdout);
        llm.printTimings();

        std::ofstream summaryFile("summaries/" + entry.path().filename().string(), std::ios::out);
        summaryFile << summary;
        summaryFile.close();
    }
}


std::string chatPrompts[] = {
    "You are Llama 3."
};

void chat() {
    ChatML llm("models/" + modelChatML, rand(), 32767, 1, 1, LLMType::CHAT, chatPrompts[0], 33, false);

    if (!llm.ok()) {
        for (auto error : llm.getErrors()) {
            std::cerr << error << "\n";
        }

        exit(1);
    }

    std::cout << "\033[2J\033[1;1H";

    while (true) {
        std::cout << "User:\n";
        std::string prompt;
        std::getline(std::cin, prompt);

        if (prompt == "exit") {
            break;
        }

        std::cout << "\nAssistant:\n";
        llm.response(prompt, 32767, true);
        //llm.printTimings();
        std::cout << "\n\n";
    }
}
