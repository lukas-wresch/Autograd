#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>



class ThreadPool
{
public:
    explicit ThreadPool(size_t numThreads) : stop(false), activeTasks(0)
    {
        m_numThreads = numThreads;

        for (size_t i = 0; i < numThreads; i++)
        {
            workers.emplace_back([this]()
            {
                while (true)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(mutex);

                        cv.wait(lock, [this]()
                        {
                            return stop || !tasks.empty();
                        });

                        if (stop && tasks.empty())
                            return;

                        task = std::move(tasks.front());
                        tasks.pop();

                        activeTasks++;
                    }

                    task();

                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        activeTasks--;

                        if (tasks.empty() && activeTasks == 0)
                            cv_done.notify_all();
                    }
                }
            });
        }
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
        }

        cv.notify_all();

        for (auto& t : workers)
            t.join();
    }

    // enqueue task
    void Enqueue(std::function<void()> task)
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            tasks.push(std::move(task));
        }

        cv.notify_one();
    }

    // wait for all tasks
    void Wait()
    {
        std::unique_lock<std::mutex> lock(mutex);

        cv_done.wait(lock, [this]()
        {
            return tasks.empty() && activeTasks == 0;
        });
    }

    size_t GetWorkerCount() const { return m_numThreads; }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable cv_done;

    bool stop;
    size_t activeTasks;
    size_t m_numThreads;
};