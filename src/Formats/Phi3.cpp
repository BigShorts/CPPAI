#include "Phi3.hpp"

std::unordered_map<std::string, int> phi3map = {
    { "<|endoftext|>", 32000 },
    { "<|assistant|>", 32001 },
    { "<|placeholder1|>", 32002 },
    { "<|placeholder2|>", 32003 },
    { "<|placeholder3|>", 32004 },
    { "<|placeholder4|>", 32005 },
    { "<|system|>", 32006 },
    { "<|end|>", 32007 },
    { "<|placeholder5|>", 32008 },
    { "<|placeholder6|>", 32009 },
    { "<|user|>", 32010 }
};

void Phi3::tokenize() {
    auto newlineToken = ::llama_tokenize(ctx, "\n", false)[1];

    tokensList = { 1 }; // bos_token
    for (auto message : messages) {
        if (message.role == "user")
            tokensList.push_back(phi3map["<|user|>"]);
        else if (message.role == "assistant")
            tokensList.push_back(phi3map["<|assistant|>"]);
        else if (message.role == "system")
            tokensList.push_back(phi3map["<|system|>"]);
        
        tokensList.push_back(newlineToken);

        auto tokens = ::llama_tokenize(ctx, message.content, false);
        tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());

        tokensList.push_back(phi3map["<|end|>"]);
        tokensList.push_back(newlineToken);
    }

    tokensList.push_back(phi3map["<|assistant|>"]);
    tokensList.push_back(newlineToken);
}

Phi3::Phi3(std::string modelPath, int seed, int nCtx, int nThreads, int nThreadsBatch, LLMType type, std::string systemMessage, int nGpuLayers, bool numa)
: LLM(modelPath, seed, nCtx, nThreads, nThreadsBatch, type, systemMessage, nGpuLayers, numa) { }
