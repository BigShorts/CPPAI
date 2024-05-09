#ifndef SRC_ARTICLES
#define SRC_ARTICLES
#include <string>
#include <sqlite3.h>

struct Article {
    int id;
    std::string title;
    std::string link;
    std::string textContent;
};

class Articles {
private:
    sqlite3* db;

public:
    Articles(std::string dbPath);
    ~Articles();

    Article getNextArticle();
    void updateSummary(int id, std::string summary);
};

#endif /* SRC_ARTICLES */
