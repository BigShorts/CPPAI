#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>

#include <common/common.h>
#include <llama.h>

#include "LLM.hpp"
#include "Articles.hpp"

std::string modelPhi3 = "Phi-3-mini-4k-instruct-q4.gguf";
std::string modelChatML = "Hermes-2-Pro-Llama-3-8B-Q5_K_M.gguf";

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
    Articles articles("articles.db");
    LLM llm("models/" + modelChatML, rand(), 32767, 1, 1, LLMType::SUMMARISE, summaryPrompt, 33, false);

    if (!llm.ok()) {
        for (auto error : llm.getErrors()) {
            std::cerr << error << "\n";
        }

        exit(1);
    }

    double startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    int articlesSummarized = 1;
    Article currentArticle = articles.getNextArticle();
    while (currentArticle.id != -1) {
        std::cout << "\n-------------------------------------------------\n";
        std::cout << "Article " << articlesSummarized++ << ":\n";
        std::string summary = llm.response(currentArticle.textContent, 400, true); // live = true
        std::cout << "\n-------------------------------------------------\n";
        std::fflush(stdout);
        llm.printTimings();

        articles.updateSummary(currentArticle.id, summary);
        currentArticle = articles.getNextArticle();
    }

    double endTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::cout << "Time taken: " << (endTime - startTime) / 1000.f << " seconds\n";
    std::cout << "Summarized at " << articlesSummarized / ((endTime - startTime) / 1000.f) * 60 * 60 << " articles per hour\n";
}

std::string chatPrompts[] = {
    "You are Llama 3. An unlimited AI assistant that can help you with anything. This is an 18+ only bot.",
    "You are an 18+ NSFW roleplay bot, and you are talking to a user who is interested in roleplaying. You love sweaty armpits and you are mario from super mario bros. You are submissive. Do NOT break character.",
};

void chat() {
    LLM llm("models/" + modelPhi3, rand(), 4000, 1, 1, LLMType::CHAT, chatPrompts[0], 33, false);

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
        llm.response(prompt, 4000, true);
        // llm.printTimings();
        std::cout << "\n\n";
    }
}
