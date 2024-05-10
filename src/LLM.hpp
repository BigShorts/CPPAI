#ifndef SRC_LLM
#define SRC_LLM
#include <vector>
#include <sstream>
#include <llama.h>
#include <common/common.h>

enum class LLMType {
    SUMMARISE,
    CHAT
};

class LLM {
protected:
    LLMType type;
    llama_context* ctx;
    llama_model* model;
    llama_batch batch;
    std::vector<llama_token> tokensList;
    std::vector<llama_chat_message> messages;
    std::vector<std::string> errors;
    std::string systemMessage = "";
    std::unordered_map<std::string, int> tokenMap;
    const int batchSize = 512;
    int nCtx;

    void decodeTokens(std::vector<llama_token> tokens);
    void decodeMessage(std::string message);

    void addMessage(std::string role, std::string content);

    void newMessagesSummarise(std::string prompt);
    void newMessagesChat(std::string prompt);

public:
    LLM(std::string modelPath, int seed, int nCtx, int nThreads, int nThreadsBatch, LLMType type, std::string systemMessage = "", int nGpuLayers = -1, bool numa = false);
    ~LLM();

    std::vector<std::string> getErrors();

    virtual std::string response(std::string prompt, int nLen, bool live = false);

    void printTimings();
    bool ok();
};

#endif /* SRC_LLM */
