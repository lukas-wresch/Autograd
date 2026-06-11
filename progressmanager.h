#pragma once

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>



class ProgressManager
{
public:
    class Handle
    {
    public:
        Handle() = default;

        ~Handle()
        {
            if (manager_)
                manager_->complete(id_);
        }

        void description(const std::string& desc)
        {
            if (!manager_)
                return;

            manager_->update(id_, desc);
        }

        void update(int current)
        {
            if (!manager_)
                return;

            manager_->update(id_, current);
        }

        void update(size_t current)
        {
            if (!manager_)
                return;

            manager_->update(id_, (int)current);
        }

    private:
        friend class ProgressManager;

        Handle(ProgressManager* manager, size_t id) : manager_(manager), id_(id)
        {
        }

        ProgressManager* manager_ = nullptr;
        size_t id_ = 0;
    };

    Handle createBar(std::string label, int total, int width = 50)
    {
        std::lock_guard lock(mutex_);

        const size_t id = nextId_++;

        bars_.emplace(id, ProgressBarData{ std::move(label), std::string(""), total, width, 0 });

        renderLocked();

        return Handle(this, id);
    }

    Handle createBar(std::string label, size_t total, int width = 50)
    {
        std::lock_guard lock(mutex_);

        const size_t id = nextId_++;

        bars_.emplace(id, ProgressBarData{ std::move(label), std::string(""), (int)total, width, 0 });

        renderLocked();

        return Handle(this, id);
    }

    void update(size_t id, int current)
    {
        std::lock_guard lock(mutex_);

        if (id >= bars_.size())
            return;

        if (bars_[id].current == current)
            return;

        bars_[id].current = current;

        renderLocked();
    }

    void update(size_t id, const std::string& desc)
    {
        std::lock_guard lock(mutex_);

        if (id >= bars_.size())
            return;

        if (bars_[id].desc == desc)
            return;

        bars_[id].desc = desc;

        renderLocked();
    }

    void complete(size_t id)
    {
        std::lock_guard lock(mutex_);
        bars_.erase(id);
        renderLocked();
    }

    ~ProgressManager()
    {
        std::lock_guard lock(mutex_);

        if (renderedLines_ > 0)
        {
            std::cout << "\033[" << renderedLines_ << "A";

            for (size_t i = 0; i < renderedLines_; i++)
            {
                std::cout << "\033[2K\r";

                if (i + 1 < renderedLines_)
                    std::cout << "\033[1B";
            }

            if (renderedLines_ > 1)
                std::cout << "\033[" << (renderedLines_ - 1) << "A";

            std::cout.flush();
        }
    }

private:
    struct ProgressBarData
    {
        std::string label;
        std::string desc;
        int total;
        int width;
        int current;
    };

    void renderLocked()
    {
        if (renderedLines_ > 0)
            std::cout << "\033[" << renderedLines_ << "A";

        for (const auto& [id, bar] : bars_)
        {
            std::cout << "\033[2K\r";
            drawBar(bar);
            std::cout << '\n';
        }

        std::cout.flush();

        renderedLines_ = bars_.size();
    }

    static void drawBar(const ProgressBarData& bar)
    {
        const float progress = bar.total > 0 ? static_cast<float>(bar.current) / bar.total : 1.0f;

        const int pos = (int)(progress * bar.width);

        std::cout << bar.label << " [";

        for (int i = 0; i < bar.width; ++i)
        {
            if (i < pos)
                std::cout << '=';
            else if (i == pos)
                std::cout << '>';
            else
                std::cout << ' ';
        }

        std::cout << "] " << (int)(progress * 100.0f) << "% " << bar.desc;
    }

private:
    std::mutex mutex_;

    std::unordered_map<size_t, ProgressBarData> bars_;
    size_t nextId_ = 0;

    size_t renderedLines_ = 0;
};