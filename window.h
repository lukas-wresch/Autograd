#pragma once
#include <vector>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "src/tensor4d.h"



class Widget
{
public:
    virtual ~Widget() = default;
    virtual void Draw(HDC hdc) = 0;
};



class ImageWidget : public Widget
{
public:
    ImageWidget(const Tensor4D& tensor, int x, int y, int scale, const std::wstring& label = L"")
        : m_Tensor(tensor), m_X(x), m_Y(y), m_Scale(scale), m_Label(label)
    {
    }

    void Draw(HDC hdc) override
    {
        const int width  = (int)(m_Tensor.GetColumns());
        const int height = (int)(m_Tensor.GetRows());

        Rectangle(
            hdc,
            m_X - 1,
            m_Y - 1,
            m_X + width * m_Scale + 1,
            m_Y + height * m_Scale + 1);

        for (int py = 0; py < height; py++)
        {
            for (int px = 0; px < width; px++)
            {
                BYTE r, g, b = 0;

                if (m_Tensor.GetDepth() == 1)
                {
                    float v = m_Tensor.At(0, 0, py, px);

                    if (v < 0.0f) v = 0.0f;
                    if (v > 1.0f) v = 1.0f;
                    //v = (rand() % 1000) * 0.001f;

                    r = g = b = (BYTE)(v * 255.0f);
                }
                else if (m_Tensor.GetDepth() == 3)
                {
                    float r_ = m_Tensor.At(0, 0, py, px);
                    float g_ = m_Tensor.At(0, 1, py, px);
                    float b_ = m_Tensor.At(0, 2, py, px);

                    if (r_ < 0.0f) r_ = 0.0f;
                    if (r_ > 1.0f) r_ = 1.0f;
                    if (g_ < 0.0f) g_ = 0.0f;
                    if (g_ > 1.0f) g_ = 1.0f;
                    if (b_ < 0.0f) b_ = 0.0f;
                    if (b_ > 1.0f) b_ = 1.0f;

                    r = (BYTE)(r_ * 255.0f);
                    g = (BYTE)(g_ * 255.0f);
                    b = (BYTE)(b_ * 255.0f);
                }

                HBRUSH brush = CreateSolidBrush(RGB(r, g, b));

                RECT rect =
                {
                    m_X + px * m_Scale,
                    m_Y + py * m_Scale,
                    m_X + (px + 1) * m_Scale,
                    m_Y + (py + 1) * m_Scale
                };

                FillRect(hdc, &rect, brush);
                DeleteObject(brush);
            }
        }

        if (!m_Label.empty())
        {
            TextOutW(hdc, m_X, m_Y + height * m_Scale + 10, m_Label.c_str(), static_cast<int>(m_Label.size()));
        }
    }

private:
    Tensor4D m_Tensor;
    int m_X;
    int m_Y;
    int m_Scale;
    std::wstring m_Label;
};



class Window
{
public:
    Window(
        const std::wstring& title,
        int width,
        int height)
        :
        m_Title(title),
        m_Width(width),
        m_Height(height)
    {}

    ~Window()
    {
        Stop();
    }

    void Show()
    {
        ShowWindow(m_hWnd, SW_SHOW);
        UpdateWindow(m_hWnd);
    }

    void RequestRedraw()
    {
        if (m_hWnd)
            InvalidateRect(m_hWnd, nullptr, FALSE);
    }

    void Start()
    {
        m_Running = true;
        m_Thread = std::thread(&Window::RunLoop, this);
    }

    void Stop()
    {
        if (!m_hWnd) return;

        m_Running = false;

        PostMessage(m_hWnd, WM_CLOSE, 0, 0);

        if (m_Thread.joinable())
            m_Thread.join();
    }

    void RunLoop()
    {
        // 1. Window class registrieren (im UI thread!)
        WNDCLASSW wc = {};
        wc.lpfnWndProc = WndProc;
        wc.hInstance = GetModuleHandle(nullptr);
        wc.lpszClassName = L"TensorViewerClass";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

        RegisterClassW(&wc);

        // 2. Window erstellen IM UI THREAD
        m_hWnd = CreateWindowExW(
            0,
            wc.lpszClassName,
            m_Title.c_str(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT,
            m_Width, m_Height,
            nullptr,
            nullptr,
            wc.hInstance,
            this
        );

        ShowWindow(m_hWnd, SW_SHOW);
        UpdateWindow(m_hWnd);

        MSG msg;

        while (m_Running && GetMessage(&msg, nullptr, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    int Run()
    {
        MSG msg;

        while (GetMessage(&msg, nullptr, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        return (int)(msg.wParam);
    }

    void AddWidget(std::unique_ptr<Widget> widget)
    {
        m_Widgets.push_back(std::move(widget));
    }

private:
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {
        Window* self = nullptr;

        if (msg == WM_NCCREATE)
        {
            CREATESTRUCT* cs = (CREATESTRUCT*)(lParam);

            self = (Window*)(cs->lpCreateParams);

            SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)(self));

            self->m_hWnd = hwnd;
        }
        else
            self = (Window*)(GetWindowLongPtr(hwnd, GWLP_USERDATA));

        if (!self)
            return DefWindowProc(hwnd, msg, wParam, lParam);

        switch (msg)
        {
        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            for (auto& widget : self->m_Widgets)
                widget->Draw(hdc);

            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        }

        return DefWindowProc(hwnd, msg, wParam, lParam);
    }

private:
    HWND m_hWnd = nullptr;

    std::wstring m_Title;
    int m_Width;
    int m_Height;

    std::vector<std::unique_ptr<Widget>> m_Widgets;

    std::thread m_Thread;
    std::atomic<bool> m_Running = false;
};