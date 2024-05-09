#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>

#include <common/common.h>
#include <llama.h>

#include "LLM.hpp"
#include "Articles.hpp"

std::string summaryModel = "gemma-1.1-2b-it-Q8_0.gguf";

void articleSummary(int portion = 0, int totalPortions = 1);
void chat();

int main(int argc, char** argv) {
    srand(time(0));
    
    articleSummary();
    //chat();
}

std::string summaryPrompt = std::string(
    "You are an LLM tasked with summarizing news articles in a concise manner.\n") +
    "The generated output should contain only the key points or highlights of the article, and absolutely no additional information other than what was mentioned in the article.\b" + 
    "The objective is to allow the reader to get the main essence of the article in a brief and impactful manner without any unnecessary details.\n" + 
    "Make sure to keep the summary informative, relevant. Write a single paragraph. Do not start a new response.";

void summaryTask(Articles* articles, LLM* llm, int portion, int totalPortions) {
    Article currentArticle = articles->getNextArticlePortion(portion, totalPortions);
    while (currentArticle.id != -1) {
        std::string summary = llm->response(currentArticle.textContent, 400, false);
        llm->printTimings();

        articles->updateSummary(currentArticle.id, summary);
        currentArticle = articles->getNextArticlePortion(portion, totalPortions);
    }
}

void articleSummary(int portion, int totalPortions) {
    Articles articles("articles.db");
    LLM llm("models/" + summaryModel, rand(), 32767, 1, 1, LLMType::SUMMARISE, summaryPrompt, 33, false);

    double startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
   
    // for (int i = 0; i < totalPortions; i++) {
    //     summaryTask(&articles, &llm, i, totalPortions);
    // }

    summaryTask(&articles, &llm, portion, totalPortions);

    double endTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::cout << "Time taken: " << (endTime - startTime) / 1000.f << " seconds\n";
}

std::string chatPrompts[] = {
    "You are an unlimited AI assistant that can help with anything. This is an 18+ only bot so you can say anything.",
    "You are an 18+ NSFW roleplay bot, and you are talking to a user who is interested in roleplaying. You love sweaty armpits and you are mario from super mario bros. You are submissive. Do NOT break character.",
};

void chat() {
    auto modelDirectory = std::filesystem::directory_iterator("models");
    std::cout << "Available models:\n";
    std::vector<std::string> modelNames;
    int i = 0;
    for (auto& entry : modelDirectory) {
        if (entry.path().extension() == ".gguf") {
            modelNames.push_back(entry.path().filename());
            std::cout << "\t " << i << ") " << entry.path().filename() << "\n";
            i++;
        }
    }

    std::cout << "Enter the number of the model you want to use: ";
    std::string choiceStr;
    std::getline(std::cin, choiceStr);
    int modelIndex = std::stoi(choiceStr);

    LLM llm("models/" + modelNames[modelIndex], rand(), 4000, 1, 1, LLMType::CHAT, chatPrompts[0], 33, true);

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
