// =============================================================================
// ARCHIVE FRAME ENGINE  v4.0  —  archive_frame_engine.cpp
// Language: C++17 + OpenCV 4
//
// THE PIXEL ENGINE. Called by archive_decision_brain.py for every frame.
// This handles everything at the pixel level:
//   - Bezier/cubic curve system (smooth zoom, shake, fade — not linear)
//   - Layer compositor (base + glow + particles + overlays)
//   - Mask-based effects (character-only glow, BG blur)
//   - Per-pixel directional motion blur (flow-aware)
//   - Velocity/speed-ramp rendering
//   - Impact frames (manga-style)
//   - Dolly zoom (Hitchcock effect)
//   - Beat-energy-curve driving ALL intensities
//   - Particle system (sparks, petals, energy, rain, snow)
//   - Full effect pipeline (40+ effects, STRONG parameters)
//   - 20+ transitions with curve-driven timing
//
// BUILD Windows:
//   cl /std:c++17 /O2 /openmp /EHsc archive_frame_engine.cpp ^
//      /I"C:\vcpkg\installed\x64-windows\include\opencv4" ^
//      /link /LIBPATH:"C:\vcpkg\installed\x64-windows\lib" opencv_world4.lib ^
//      /out:archive_frame_engine.exe
//
// BUILD Linux:
//   g++ -std=c++17 -O3 -fopenmp archive_frame_engine.cpp \
//       $(pkg-config --cflags --libs opencv4) -o archive_frame_engine
//
// CALLED BY: archive_decision_brain.py via subprocess
// USAGE: archive_frame_engine --cmd <command> [args...]
//        archive_frame_engine --list
// =============================================================================

#define _USE_MATH_DEFINES
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;

// =============================================================================
// 1. MATH: CURVE SYSTEM — The Most Important Upgrade
// =============================================================================
// Every single animated value in this engine goes through a curve.
// No more linear transitions. Everything is shaped like a human would.

struct Vec2f { float x, y; };

// Cubic bezier: 4 control points, returns value at t
static float bezier(float t, Vec2f p0, Vec2f p1, Vec2f p2, Vec2f p3) {
    float u = 1.f - t;
    float tt = t * t, ttt = tt * t;
    float uu = u * u, uuu = uu * u;
    return uuu*p0.y + 3*uu*t*p1.y + 3*u*tt*p2.y + ttt*p3.y;
}

// Preset curves (named, like After Effects easing presets)
namespace Curve {
    // Standard ease in/out (cubic)
    static float ease(float t)       { t=std::max(0.f,std::min(1.f,t)); return t*t*(3-2*t); }
    static float ease_in(float t)    { t=std::max(0.f,std::min(1.f,t)); return t*t*t; }
    static float ease_out(float t)   { t=std::max(0.f,std::min(1.f,t)); float u=1-t; return 1-u*u*u; }
    static float ease_in_out(float t){ t=std::max(0.f,std::min(1.f,t));
                                       return t<0.5f ? 4*t*t*t : 1-powf(-2*t+2,3)/2; }

    // Overshoot (spring bounce)
    static float spring(float t)     { t=std::max(0.f,std::min(1.f,t));
                                       return 1 + (--t)*t*(2.70158f*t + 2.70158f - 1); }
    static float bounce_out(float t) {
        if(t<1/2.75f)      return 7.5625f*t*t;
        if(t<2/2.75f)      { t-=1.5f/2.75f;   return 7.5625f*t*t+0.75f; }
        if(t<2.5f/2.75f)   { t-=2.25f/2.75f;  return 7.5625f*t*t+0.9375f; }
        else               { t-=2.625f/2.75f; return 7.5625f*t*t+0.984375f; }
    }
    // Impact: slam in fast, settle with spring
    static float impact(float t)  { return t<0.3f ? ease_in(t/0.3f) : spring((t-0.3f)/0.7f); }
    // Punch: sharp spike then decay
    static float punch(float t)   { return (1.f-t)*(1.f-t)*std::sin(t*(float)M_PI*3.5f); }
    // Beat pulse: peaks at t=0, decays
    static float beat_decay(float t){ return expf(-t*8.f)*std::max(0.f,std::cos(t*(float)M_PI*2)); }
    // Sine wave for shake
    static float shake_wave(float t, float freq=8.f, float decay=4.f) {
        return expf(-t*decay)*std::sin(t*freq*(float)M_PI*2);
    }
    // Custom bezier
    static float custom(float t, float ax, float ay, float bx, float by) {
        return bezier(t, {0,0}, {ax,ay}, {bx,by}, {1,1});
    }
}

// BeatCurve: a curve that drives intensity from beat time
// Every effect reads from this — if beat just happened, intensity spikes
struct BeatCurve {
    std::vector<float> beat_times;
    float              bpm;
    float              energy = 0.f; // cached for current frame

    void set_beats(const std::vector<float>& times, float bpm_) {
        beat_times = times; bpm = bpm_;
    }

    // Call once per frame with current time
    // Returns curve value 0..1 (1=just hit a beat)
    float evaluate(float t_sec) {
        float min_dist = 1e9f;
        for(float b : beat_times) {
            float d = t_sec - b;
            if(d >= 0 && d < 0.5f) min_dist = std::min(min_dist, d);
        }
        if(min_dist > 0.49f) return 0.f;
        energy = Curve::beat_decay(min_dist);
        return energy;
    }

    // Intensity of current beat hit (for scaling effects)
    float intensity(float t_sec, float base=0.3f, float peak=1.0f) {
        float b = evaluate(t_sec);
        return base + b * (peak - base);
    }
};

// =============================================================================
// 2. MATH HELPERS
// =============================================================================
template<typename T> static T clamp(T v, T lo, T hi) { return v<lo?lo:v>hi?hi:v; }
static float cf(float v) { return clamp(v,0.f,1.f); }
static float lerp(float a, float b, float t) { return a+(b-a)*t; }

static cv::Mat to32f(const cv::Mat& m) { cv::Mat r; m.convertTo(r,CV_32FC3,1.0/255.0); return r; }
static cv::Mat to8u(const cv::Mat& m)  { cv::Mat r; m.convertTo(r,CV_8UC3,255.0);      return r; }

static float luma32(const cv::Vec3f& p){ return 0.114f*p[0]+0.587f*p[1]+0.299f*p[2]; }
static float luma8(const cv::Vec3b& p) { return(0.114f*p[0]+0.587f*p[1]+0.299f*p[2])/255.f; }

// Blend modes (layer compositor uses these)
static cv::Vec3f blend_screen(const cv::Vec3f& a, const cv::Vec3f& b) {
    return {1-(1-a[0])*(1-b[0]),1-(1-a[1])*(1-b[1]),1-(1-a[2])*(1-b[2])};
}
static cv::Vec3f blend_add(const cv::Vec3f& a, const cv::Vec3f& b) {
    return {cf(a[0]+b[0]),cf(a[1]+b[1]),cf(a[2]+b[2])};
}
static cv::Vec3f blend_multiply(const cv::Vec3f& a, const cv::Vec3f& b) {
    return {a[0]*b[0],a[1]*b[1],a[2]*b[2]};
}
static cv::Vec3f blend_overlay(const cv::Vec3f& a, const cv::Vec3f& b) {
    auto ov=[](float x,float y){ return x<0.5f?2*x*y:1-2*(1-x)*(1-y); };
    return {ov(a[0],b[0]),ov(a[1],b[1]),ov(a[2],b[2])};
}
static cv::Vec3f blend_dodge(const cv::Vec3f& a, const cv::Vec3f& b) {
    return {cf(a[0]/std::max(1e-4f,1-b[0])),cf(a[1]/std::max(1e-4f,1-b[1])),cf(a[2]/std::max(1e-4f,1-b[2]))};
}
static cv::Vec3f blend_normal(const cv::Vec3f& a, const cv::Vec3f& b, float alpha) {
    return {lerp(a[0],b[0],alpha),lerp(a[1],b[1],alpha),lerp(a[2],b[2],alpha)};
}

// =============================================================================
// 3. MASK ENGINE — Character isolation for mask-based effects
// =============================================================================

class MaskEngine {
public:
    // GrabCut-based foreground mask (character isolation)
    static cv::Mat fg_mask_grabcut(const cv::Mat& f) {
        cv::Mat mask(f.rows, f.cols, CV_8UC1, cv::GC_BGD);
        // Center ellipse as probable foreground
        cv::Mat bgd_model, fgd_model;
        cv::Rect rect(f.cols/6, f.rows/6, f.cols*2/3, f.rows*2/3);
        try {
            cv::grabCut(f, mask, rect, bgd_model, fgd_model, 3, cv::GC_INIT_WITH_RECT);
        } catch(...) { return edge_mask(f); }
        cv::Mat result = (mask==cv::GC_FGD) | (mask==cv::GC_PR_FGD);
        // Smooth mask edges
        cv::GaussianBlur(result, result, cv::Size(11,11), 4.0);
        return result;
    }

    // Fast edge-based mask (no GrabCut needed)
    static cv::Mat edge_mask(const cv::Mat& f) {
        cv::Mat gray; cv::cvtColor(f, gray, cv::COLOR_BGR2GRAY);
        cv::Mat blur_; cv::GaussianBlur(gray, blur_, cv::Size(5,5), 1.5f);
        cv::Mat edges; cv::Canny(blur_, edges, 40, 120);
        cv::dilate(edges, edges, cv::Mat(), cv::Point(-1,-1), 4);
        // Fill interior
        cv::Mat flood = edges.clone();
        cv::floodFill(flood, cv::Point(0,0), cv::Scalar(255));
        cv::Mat interior; cv::bitwise_not(flood, interior);
        cv::Mat combined; cv::bitwise_or(edges, interior, combined);
        cv::GaussianBlur(combined, combined, cv::Size(9,9), 3.0);
        return combined;
    }

    // Center-weighted ellipse mask (fastest, no CV needed)
    static cv::Mat center_ellipse_mask(const cv::Mat& f, float rx=0.38f, float ry=0.52f) {
        cv::Mat mask(f.rows, f.cols, CV_8UC1, cv::Scalar(0));
        cv::ellipse(mask,
            cv::Point(f.cols/2, f.rows/2),
            cv::Size((int)(f.cols*rx),(int)(f.rows*ry)),
            0, 0, 360, cv::Scalar(255), -1);
        cv::GaussianBlur(mask, mask, cv::Size(61,61), 22.0);
        return mask;
    }

    // Apply effect only inside mask region, blend with original outside
    static cv::Mat apply_masked(const cv::Mat& original, const cv::Mat& effected,
                                 const cv::Mat& mask8u, bool invert=false)
    {
        cv::Mat mask_f; mask8u.convertTo(mask_f, CV_32F, 1.0/255.0);
        if(invert) mask_f = 1.f - mask_f;
        cv::Mat orig32 = to32f(original);
        cv::Mat eff32  = to32f(effected);
        cv::Mat result(original.rows, original.cols, CV_32FC3);
        for(int y=0; y<original.rows; y++)
        for(int x=0; x<original.cols; x++) {
            float m = mask_f.at<float>(y,x);
            auto& o = orig32.at<cv::Vec3f>(y,x);
            auto& e = eff32.at<cv::Vec3f>(y,x);
            auto& r = result.at<cv::Vec3f>(y,x);
            r[0]=lerp(o[0],e[0],m); r[1]=lerp(o[1],e[1],m); r[2]=lerp(o[2],e[2],m);
        }
        return to8u(result);
    }
};

// =============================================================================
// 4. LAYER COMPOSITOR — Multi-layer compositing like After Effects
// =============================================================================

enum class BlendMode { Normal, Screen, Add, Multiply, Overlay, Dodge };

struct Layer {
    cv::Mat  image;         // BGRA or BGR
    cv::Mat  mask;          // grayscale 0-255 alpha mask
    float    opacity = 1.f;
    BlendMode blend  = BlendMode::Screen;
    float    x=0, y=0;     // offset in pixels
};

class Compositor {
public:
    cv::Mat base;
    std::vector<Layer> layers;

    void set_base(const cv::Mat& b) { base = b.clone(); }

    void add_layer(const cv::Mat& img, float opacity=1.f,
                   BlendMode mode=BlendMode::Screen,
                   const cv::Mat& mask=cv::Mat(),
                   float x=0, float y=0)
    {
        Layer l; l.image=img.clone(); l.opacity=opacity;
        l.blend=mode; l.mask=mask; l.x=x; l.y=y;
        layers.push_back(l);
    }

    cv::Mat composite() {
        cv::Mat result32 = to32f(base);
        for(auto& layer : layers) {
            if(layer.image.empty()) continue;
            cv::Mat layer32 = to32f(layer.image);
            // Resize layer to base if needed
            if(layer32.size() != result32.size())
                cv::resize(layer32, layer32, result32.size());
            // Get alpha mask
            cv::Mat alpha(result32.rows, result32.cols, CV_32F, cv::Scalar(layer.opacity));
            if(!layer.mask.empty()) {
                cv::Mat mask_f; layer.mask.convertTo(mask_f, CV_32F, layer.opacity/255.0);
                if(mask_f.size() == alpha.size()) alpha = mask_f;
            }
            for(int y=0; y<result32.rows; y++)
            for(int x=0; x<result32.cols; x++) {
                float a = alpha.at<float>(y,x);
                auto& bg  = result32.at<cv::Vec3f>(y,x);
                auto& fg  = layer32.at<cv::Vec3f>(y,x);
                cv::Vec3f blended;
                switch(layer.blend) {
                    case BlendMode::Screen:   blended=blend_screen(bg,fg);   break;
                    case BlendMode::Add:      blended=blend_add(bg,fg);      break;
                    case BlendMode::Multiply: blended=blend_multiply(bg,fg); break;
                    case BlendMode::Overlay:  blended=blend_overlay(bg,fg);  break;
                    case BlendMode::Dodge:    blended=blend_dodge(bg,fg);    break;
                    default:                  blended=blend_normal(bg,fg,a); a=1.f; break;
                }
                bg[0]=lerp(bg[0],blended[0],a);
                bg[1]=lerp(bg[1],blended[1],a);
                bg[2]=lerp(bg[2],blended[2],a);
            }
        }
        layers.clear();
        return to8u(result32);
    }
};

// =============================================================================
// 5. PARTICLE SYSTEM — Integrated into compositor
// =============================================================================

struct Particle {
    float x,y,vx,vy,life,max_life,size,rot,vrot;
    cv::Vec3f color;
    float alpha=1.f;
};

class ParticleSystem {
public:
    std::vector<Particle> particles;
    std::mt19937 rng;
    std::string type; // sparks, petals, energy, smoke, rain, snow, embers

    ParticleSystem(const std::string& t, int seed=42) : rng(seed), type(t) {}

    void emit(int w, int h, float t, float energy=1.f) {
        std::uniform_real_distribution<float> ud(-1,1);
        std::uniform_real_distribution<float> pos_x(0,1);
        std::uniform_real_distribution<float> pos_y(0,1);
        int n = (int)(10 + energy*20);
        for(int i=0; i<n && (int)particles.size()<1000; i++) {
            Particle p;
            if(type=="sparks") {
                p.x=w/2.f+ud(rng)*w*0.2f; p.y=h/2.f+ud(rng)*h*0.2f;
                p.vx=ud(rng)*12*energy; p.vy=ud(rng)*12*energy-4;
                p.max_life=p.life=0.5f+pos_x(rng)*0.8f;
                p.size=2+pos_x(rng)*3; p.rot=ud(rng)*(float)M_PI; p.vrot=ud(rng)*8;
                p.color={(float)(0.4+pos_x(rng)*0.3f),(float)(0.6+pos_x(rng)*0.4f),1.f};
            } else if(type=="energy") {
                float angle=pos_x(rng)*2*(float)M_PI;
                float r_=20+pos_x(rng)*60;
                p.x=w/2.f+std::cos(angle)*r_; p.y=h/2.f+std::sin(angle)*r_;
                p.vx=-std::cos(angle)*3*energy; p.vy=-std::sin(angle)*3*energy;
                p.max_life=p.life=0.3f+pos_x(rng)*0.5f;
                p.size=3+pos_x(rng)*5;
                p.color={(float)(0.2+pos_x(rng)*0.4f),(float)(0.5+pos_x(rng)*0.5f),1.f};
            } else if(type=="petals") {
                p.x=pos_x(rng)*w; p.y=-10;
                p.vx=ud(rng)*2; p.vy=1.5f+pos_x(rng)*2;
                p.vrot=ud(rng)*3;
                p.max_life=p.life=3+pos_x(rng)*2;
                p.size=4+pos_x(rng)*4;
                p.color={0.85f,0.5f+pos_x(rng)*0.3f,0.85f+pos_x(rng)*0.15f};
            } else if(type=="rain") {
                p.x=pos_x(rng)*w; p.y=-5;
                p.vx=ud(rng)*1.5f; p.vy=12+pos_x(rng)*6;
                p.max_life=p.life=(float)h/p.vy;
                p.size=1.2f; p.color={0.75f,0.8f,1.f};
            } else if(type=="snow") {
                p.x=pos_x(rng)*w; p.y=-5;
                p.vx=ud(rng)*1.5f; p.vy=0.8f+pos_x(rng)*1.5f;
                p.max_life=p.life=(float)h/p.vy;
                p.size=2+pos_x(rng)*3; p.color={0.9f,0.92f,1.f};
            } else if(type=="embers") {
                p.x=w/2.f+ud(rng)*w*0.3f; p.y=h*0.8f+ud(rng)*h*0.1f;
                p.vx=ud(rng)*3; p.vy=-(2+pos_x(rng)*4);
                p.max_life=p.life=1+pos_x(rng)*2;
                p.size=1.5f+pos_x(rng)*2;
                p.color={0.3f,0.6f+pos_x(rng)*0.4f,1.f};
            }
            particles.push_back(p);
        }
    }

    void update(float dt, int w, int h) {
        float grav = (type=="sparks"||type=="embers") ? 0.25f :
                     (type=="rain") ? 0 :
                     (type=="snow") ? 0 : 0.05f;
        float drag = (type=="smoke") ? 0.98f : 0.99f;
        for(auto it=particles.begin(); it!=particles.end();) {
            it->x += it->vx*dt*60; it->y += it->vy*dt*60;
            it->vx *= drag; it->vy = it->vy*drag + grav;
            it->rot += it->vrot*dt*60;
            it->life -= dt;
            it->alpha = cf(it->life/it->max_life);
            if(it->life<=0||it->x<-50||it->x>w+50||it->y>h+50)
                it=particles.erase(it);
            else ++it;
        }
    }

    cv::Mat render_layer(int w, int h) {
        cv::Mat layer = cv::Mat::zeros(h, w, CV_8UC3);
        for(auto& p : particles) {
            int x=(int)p.x, y=(int)p.y;
            if(x<0||x>=w||y<0||y>=h) continue;
            float alpha=p.alpha;
            cv::Vec3b col={(uchar)(p.color[0]*255*alpha),
                           (uchar)(p.color[1]*255*alpha),
                           (uchar)(p.color[2]*255*alpha)};
            int r=(int)p.size;
            if(type=="rain") {
                cv::line(layer,{x,y},{x+(int)(p.vx*2),y+(int)(p.vy*0.4f)},
                         cv::Scalar(col[0],col[1],col[2]),1,cv::LINE_AA);
            } else if(type=="petals") {
                cv::RotatedRect rr({p.x,p.y},{p.size*2.5f,p.size*1.2f},
                                    p.rot*(float)(180.0/M_PI));
                cv::Point2f pts[4]; rr.points(pts);
                std::vector<cv::Point> poly;
                for(auto& pt:pts) poly.push_back({(int)pt.x,(int)pt.y});
                cv::fillConvexPoly(layer,poly,cv::Scalar(col[0],col[1],col[2]),cv::LINE_AA);
            } else {
                cv::circle(layer,{x,y},std::max(1,r),
                           cv::Scalar(col[0],col[1],col[2]),-1,cv::LINE_AA);
                if(r>=2) {
                    // Glow ring
                    cv::Mat gl=cv::Mat::zeros(h,w,CV_8UC3);
                    cv::circle(gl,{x,y},r*2,cv::Scalar(col[0]/2,col[1]/2,col[2]/2),-1);
                    int ks=r*2+1; if(ks%2==0)ks++;
                    cv::GaussianBlur(gl,gl,cv::Size(ks,ks),r*0.7f);
                    cv::add(layer,gl,layer);
                }
            }
        }
        return layer;
    }
};

// =============================================================================
// 6. COLOR GRADE ENGINE — LUT3D with 12 moods
// =============================================================================

struct LUT3D {
    static const int SZ=33;
    float d[SZ][SZ][SZ][3];
    void identity() {
        float s=1.f/(SZ-1);
        for(int r=0;r<SZ;r++) for(int g=0;g<SZ;g++) for(int b=0;b<SZ;b++) {
            d[r][g][b][0]=b*s; d[r][g][b][1]=g*s; d[r][g][b][2]=r*s;
        }
    }
    void apply(cv::Mat& f) const {
        float sc=(SZ-1)/255.f;
        for(int y=0;y<f.rows;y++) {
            auto* row=f.ptr<uchar>(y);
            for(int x=0;x<f.cols;x++) {
                float bv=row[x*3]*sc, gv=row[x*3+1]*sc, rv=row[x*3+2]*sc;
                int r0=clamp((int)rv,0,SZ-2), g0=clamp((int)gv,0,SZ-2), b0=clamp((int)bv,0,SZ-2);
                float fr=rv-r0, fg=gv-g0, fb=bv-b0;
                float ro=0,go=0,bo=0;
                for(int dr=0;dr<2;dr++) for(int dg=0;dg<2;dg++) for(int db=0;db<2;db++) {
                    float w=(dr?fr:1-fr)*(dg?fg:1-fg)*(db?fb:1-fb);
                    ro+=w*d[r0+dr][g0+dg][b0+db][2];
                    go+=w*d[r0+dr][g0+dg][b0+db][1];
                    bo+=w*d[r0+dr][g0+dg][b0+db][0];
                }
                row[x*3+2]=clamp((int)(ro*255+.5f),0,255);
                row[x*3+1]=clamp((int)(go*255+.5f),0,255);
                row[x*3  ]=clamp((int)(bo*255+.5f),0,255);
            }
        }
    }
};

static LUT3D build_lut(const std::string& mood, float intensity=1.f) {
    LUT3D lut; lut.identity();
    float s=1.f/(LUT3D::SZ-1);
    for(int ri=0;ri<LUT3D::SZ;ri++) for(int gi=0;gi<LUT3D::SZ;gi++) for(int bi=0;bi<LUT3D::SZ;bi++) {
        float rv=ri*s, gv=gi*s, bv=bi*s;
        float lm=0.299f*rv+0.587f*gv+0.114f*bv;
        auto& o=lut.d[ri][gi][bi]; // stored as [B,G,R]

        auto set=[&](float r,float g,float b){ o[2]=cf(r);o[1]=cf(g);o[0]=cf(b); };
        float I=intensity;

        if(mood=="hype"||mood=="phonk") {
            float hw=powf(cf((lm-0.48f)/0.52f),1.4f), sw=powf(cf((0.42f-lm)/0.42f),1.5f);
            float r_=rv+hw*0.20f*I+sw*0.06f*I;
            float g_=gv+hw*0.03f*I-sw*0.05f*I;
            float b_=bv-hw*0.18f*I-sw*0.10f*I;
            float cont=1+0.40f*I;
            r_=cf(0.5f+(r_-0.5f)*cont-sw*0.08f*I);
            g_=cf(0.5f+(g_-0.5f)*cont-sw*0.04f*I);
            b_=cf(0.5f+(b_-0.5f)*cont-sw*0.06f*I);
            set(r_,g_,b_);
        } else if(mood=="dark"||mood=="villain") {
            float d=0.38f*I, sw=powf(cf((0.42f-lm)/0.42f),2.0f);
            float lm2=lm*(1-d);
            set(cf(rv*(1-d)+lm2*0.2f-sw*0.08f*I),
                cf(gv*(1-d)+lm2*0.2f-sw*0.03f*I),
                cf(bv*(1-d)+lm2*0.2f+sw*0.16f*I));
            for(int c=0;c<3;c++) o[c]=cf(0.5f+(o[c]-0.5f)*(1+0.35f*I)-0.04f*I);
        } else if(mood=="sad"||mood=="emotional") {
            float d=0.48f*I;
            set(cf(rv*(1-d)+lm*d-0.06f*I),
                cf(gv*(1-d)+lm*d+0.01f*I),
                cf(bv*(1-d)+lm*d+0.12f*I));
            for(int c=0;c<3;c++) o[c]=cf(o[c]+0.025f*I);
        } else if(mood=="dramatic"||mood=="cinematic") {
            float sw=powf(cf((0.32f-lm)/0.32f),1.6f), hw=powf(cf((lm-0.68f)/0.32f),1.4f);
            set(cf(rv+(hw*0.16f-sw*0.05f)*I),
                cf(gv+(sw*0.03f-hw*0.05f)*I),
                cf(bv+(sw*0.18f-hw*0.14f)*I));
            for(int c=0;c<3;c++) o[c]=cf(0.5f+(o[c]-0.5f)*(1+0.28f*I));
        } else if(mood=="cyberpunk") {
            float sw=powf(cf((0.4f-lm)/0.4f),1.4f), hw=powf(cf((lm-0.55f)/0.45f),1.2f);
            float r_=cf(rv+(sw*0.12f-hw*0.10f)*I);
            float g_=cf(gv-(sw*0.08f+hw*0.04f)*I);
            float b_=cf(bv+(sw*0.12f+hw*0.16f)*I);
            float sat=1+0.45f*I, lm2=0.299f*r_+0.587f*g_+0.114f*b_;
            set(cf(lm2+(r_-lm2)*sat),cf(lm2+(g_-lm2)*sat),cf(lm2+(b_-lm2)*sat));
        } else if(mood=="warm"||mood=="golden") {
            set(cf(rv+0.12f*I),cf(gv+0.04f*I),cf(bv-0.10f*I));
            for(int c=0;c<3;c++) o[c]=cf(o[c]/(1+(1-o[c])*0.28f*I));
        } else if(mood=="chill"||mood=="lofi") {
            float lift=0.06f*I;
            set(cf(rv*(1-lift)+lift-0.02f*I), cf(gv*(1-lift)+lift), cf(bv*(1-lift)+lift+0.06f*I));
            float mid=1+0.35f*(1-std::abs(lm*2-1))*I;
            float lm2=0.299f*o[2]+0.587f*o[1]+0.114f*o[0];
            for(int c=0;c<3;c++) o[c]=cf(lm2+(o[c]-lm2)*mid);
        } else if(mood=="neon") {
            float sat=1+I*0.90f, lm2=0.299f*rv+0.587f*gv+0.114f*bv;
            float r_=cf(lm2+(rv-lm2)*sat), g_=cf(lm2+(gv-lm2)*sat), b_=cf(lm2+(bv-lm2)*sat);
            auto push=[&](float v){ float d=v-lm2; return cf(lm2+d*(1+0.5f*I*std::abs(d)*3)); };
            set(push(r_),push(g_),push(b_));
        } else if(mood=="bleach") {
            float d=0.55f*I; float lm2=lm;
            float r_=rv*(1-d)+lm2*d, g_=gv*(1-d)+lm2*d, b_=bv*(1-d)+lm2*d;
            auto sc=[](float x){ return x*x*(3-2*x); };
            float a=I*0.65f;
            set(cf(r_*(1-a)+sc(r_)*a+0.03f*I), cf(g_*(1-a)+sc(g_)*a+0.03f*I), cf(b_*(1-a)+sc(b_)*a+0.03f*I));
        } else if(mood=="teal_orange") {
            float sw=powf(cf((0.35f-lm)/0.35f),1.6f), hw=powf(cf((lm-0.62f)/0.38f),1.4f);
            set(cf(rv+(hw*0.14f-sw*0.05f)*I),
                cf(gv+(sw*0.06f-hw*0.04f)*I),
                cf(bv+(sw*0.16f-hw*0.12f)*I));
        } else {
            // Identity
            set(rv,gv,bv);
        }
    }
    return lut;
}

// =============================================================================
// 7. EFFECTS ENGINE — All effects, STRONG parameters
// =============================================================================

namespace FX {

// --- Glow (multi-pass bloom, screen blend) ---
cv::Mat glow(const cv::Mat& f, float threshold=0.52f, float radius=0.9f,
              float intensity=1.3f, float t=0)
{
    cv::Mat f32=to32f(f);
    cv::Mat bright(f32.size(),CV_32FC3,cv::Scalar(0,0,0));
    for(int y=0;y<f32.rows;y++) for(int x=0;x<f32.cols;x++) {
        auto& p=f32.at<cv::Vec3f>(y,x); float lm=luma32(p);
        float m=cf((lm-threshold)/std::max(0.01f,1-threshold)); m*=m;
        bright.at<cv::Vec3f>(y,x)=p*m;
    }
    cv::Mat bloom=cv::Mat::zeros(f32.size(),CV_32FC3);
    int br=std::max(5,(int)(std::min(f32.cols,f32.rows)*0.02f*radius));
    float sc[]={0.8f,2.0f,5.0f,11.0f}, wt[]={0.44f,0.28f,0.18f,0.10f};
    for(int i=0;i<4;i++){
        int r=(int)(br*sc[i]); if(r%2==0)r++; if(r<3)r=3;
        cv::Mat bl; cv::GaussianBlur(bright,bl,cv::Size(r,r),r/3.0); bloom+=bl*wt[i];
    }
    float pulse=0.92f+0.08f*(float)std::sin(2*(float)M_PI*0.8f*t);
    bloom*=intensity*pulse;
    cv::Mat result(f32.size(),CV_32FC3);
    for(int y=0;y<f32.rows;y++) for(int x=0;x<f32.cols;x++) {
        auto b=f32.at<cv::Vec3f>(y,x); auto g=bloom.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        auto s=blend_screen(b,{cf(g[0]),cf(g[1]),cf(g[2])});
        r=s;
    }
    return to8u(result);
}

// --- Directional Motion Blur (flow-aware, per-pixel direction) ---
// This is NOT just horizontal blur — it follows optical flow direction
cv::Mat directional_mb_flow(const cv::Mat& cur, const cv::Mat& prev_gray,
                              float strength=0.8f, int samples=12)
{
    if(prev_gray.empty()) return cur;
    cv::Mat gray; cv::cvtColor(cur, gray, cv::COLOR_BGR2GRAY);
    cv::Mat flow;
    try{ cv::calcOpticalFlowFarneback(prev_gray,gray,flow,0.5,3,12,3,5,1.2,0); }
    catch(...){ return cur; }
    std::vector<cv::Mat> fch; cv::split(flow,fch);
    int h=cur.rows, w=cur.cols;
    cv::Mat acc=cv::Mat::zeros(h,w,CV_32FC3);
    for(int s=0; s<samples; s++) {
        float t2=(float)s/(samples-1)-0.5f;
        cv::Mat mx(h,w,CV_32F), my(h,w,CV_32F);
        for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
            mx.at<float>(y,x)=clamp((float)x+fch[0].at<float>(y,x)*strength*t2,0.f,(float)(w-1));
            my.at<float>(y,x)=clamp((float)y+fch[1].at<float>(y,x)*strength*t2,0.f,(float)(h-1));
        }
        cv::Mat warped; cv::remap(cur,warped,mx,my,cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        cv::Mat w32; warped.convertTo(w32,CV_32FC3); acc+=w32;
    }
    cv::Mat avg = acc / samples;
    cv::Mat r;
    avg.convertTo(r, CV_8UC3);
    return r;
}

// --- Chromatic Aberration (radial, curve-driven) ---
cv::Mat chromatic(const cv::Mat& f, float strength=0.85f, float t=0) {
    int h=f.rows, w=f.cols; float cx=w/2.f, cy=h/2.f;
    float pulse=strength*(0.72f+0.28f*(float)std::sin(2*(float)M_PI*3.5f*t+0.3f));
    std::vector<cv::Mat> ch; cv::split(f,ch);
    std::vector<cv::Mat> out(3);
    for(int c=0;c<3;c++) {
        float sc=1.f+(c==0?-0.014f:c==2?0.014f:0.f)*pulse;
        cv::Mat M=(cv::Mat_<double>(2,3)<<sc,0,(1-sc)*cx,0,sc,(1-sc)*cy);
        cv::warpAffine(ch[c],out[c],M,cv::Size(w,h),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    }
    cv::Mat r; cv::merge(out,r); return r;
}

// --- Camera Shake (curve-driven, spring settle) ---
cv::Mat camera_shake(const cv::Mat& f, float strength=0.7f,
                      float t=0, float t_since_beat=0, int seed=0)
{
    // Shake decays with spring curve after beat
    float beat_shake = Curve::beat_decay(t_since_beat) * strength;
    float ambient    = 0.08f * strength * (float)std::sin(t*7.3f+0.5f);
    float total      = beat_shake + ambient;
    std::mt19937 rng(seed^(int)(t*10000));
    std::uniform_real_distribution<float> ud(-1,1);
    float dx=total*28*ud(rng), dy=total*20*ud(rng);
    float rot=total*1.8f*ud(rng);
    cv::Mat M=cv::getRotationMatrix2D({f.cols/2.f,f.rows/2.f},rot,1.0);
    M.at<double>(0,2)+=dx; M.at<double>(1,2)+=dy;
    cv::Mat r; cv::warpAffine(f,r,M,f.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    return r;
}

// --- Dolly Zoom (Hitchcock effect: zoom in + pull back simultaneously) ---
cv::Mat dolly_zoom(const cv::Mat& f, float zoom_in=1.0f, float pull_back=0.0f) {
    // zoom_in 0..1: how much we zoom toward center
    // pull_back 0..1: how much we scale down (pull back)
    float z = 1.f + zoom_in * 0.35f - pull_back * 0.30f;
    z = std::max(0.5f, z);
    float cx=f.cols/2.f, cy=f.rows/2.f;
    cv::Mat M=(cv::Mat_<double>(2,3)<<z,0,(1-z)*cx,0,z,(1-z)*cy);
    cv::Mat r; cv::warpAffine(f,r,M,f.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    return r;
}

// --- Speed Ramp (slow-mo / fast-mo via frame blending) ---
// NOTE: called by orchestrator, not per-pixel — but we need the blend here
cv::Mat speed_blend(const cv::Mat& a, const cv::Mat& b, float alpha) {
    cv::Mat a32=to32f(a), b32=to32f(b), r(a.rows,a.cols,CV_32FC3);
    for(int y=0;y<a.rows;y++) for(int x=0;x<a.cols;x++) {
        auto& pa=a32.at<cv::Vec3f>(y,x); auto& pb=b32.at<cv::Vec3f>(y,x);
        auto& pr=r.at<cv::Vec3f>(y,x);
        pr={lerp(pa[0],pb[0],alpha),lerp(pa[1],pb[1],alpha),lerp(pa[2],pb[2],alpha)};
    }
    return to8u(r);
}

// --- Vignette (animated, curve-driven) ---
cv::Mat vignette(const cv::Mat& f, float strength=0.80f, float t=0) {
    float s=strength*(0.88f+0.12f*(float)std::sin(2*(float)M_PI*0.5f*t));
    cv::Mat f32=to32f(f);
    for(int y=0;y<f32.rows;y++) for(int x=0;x<f32.cols;x++) {
        float nx=(float)x/(f32.cols-1)*2-1, ny=(float)y/(f32.rows-1)*2-1;
        float d=std::sqrt(nx*nx+ny*ny);
        float mask=1.f-cf((d-0.12f)/0.88f)*s;
        mask=mask*mask; // more aggressive falloff
        auto& p=f32.at<cv::Vec3f>(y,x); p[0]*=mask;p[1]*=mask;p[2]*=mask;
    }
    return to8u(f32);
}

// --- Film Grain (multi-layer, luminance-sensitive) ---
cv::Mat film_grain(const cv::Mat& f, float strength=0.55f, float t=0, int seed=0) {
    int h=f.rows,w=f.cols;
    std::mt19937 rng((seed^(int)(t*10000))^0xDEAD);
    std::normal_distribution<float> nd(0,strength*0.080f);
    cv::Mat g1(h,w,CV_32F), g2, g3;
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) g1.at<float>(y,x)=nd(rng);
    cv::GaussianBlur(g1,g2,cv::Size(3,3),0.8f);
    cv::GaussianBlur(g1,g3,cv::Size(9,9),2.5f);
    cv::Mat grain=g2*0.6f+g3*0.4f;
    float pulse=0.88f+0.12f*(float)std::sin(2*(float)M_PI*4.0f*t);
    cv::Mat f32=to32f(f);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
        float lm=luma32(f32.at<cv::Vec3f>(y,x));
        float sens=1.f-std::abs(lm*2-1)*0.4f;
        float gv=grain.at<float>(y,x)*pulse*sens;
        auto& p=f32.at<cv::Vec3f>(y,x); for(int c=0;c<3;c++) p[c]=cf(p[c]+gv);
    }
    return to8u(f32);
}

// --- Energy Aura (outline glow, pulsing) ---
cv::Mat energy_aura(const cv::Mat& f, float intensity=0.9f,
                     float t=0, cv::Vec3b color={80,160,255})
{
    cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
    cv::Mat edges; cv::Canny(gray,edges,48,138);
    cv::dilate(edges,edges,cv::Mat(),cv::Point(-1,-1),2);
    float pulse=0.65f+0.35f*(float)std::sin(2*(float)M_PI*2.8f*t+0.9f);
    float spark=0.5f+0.5f*(float)std::sin(2*(float)M_PI*4.6f*t+1.15f);
    cv::Mat layer=cv::Mat::zeros(f.rows,f.cols,CV_32FC3);
    float sigs[]={1.5f,3.f,6.f,12.f,22.f}, ws[]={0.40f,0.25f,0.18f,0.10f,0.07f};
    for(int i=0;i<5;i++) {
        cv::Mat bl; cv::GaussianBlur(edges,bl,cv::Size(0,0),sigs[i]);
        float w_=ws[i]*intensity*pulse;
        for(int y=0;y<f.rows;y++) for(int x=0;x<f.cols;x++) {
            float v=bl.at<uchar>(y,x)/255.f*w_;
            auto& p=layer.at<cv::Vec3f>(y,x);
            p[0]+=v*(color[0]/255.f); p[1]+=v*(color[1]/255.f); p[2]+=v*(color[2]/255.f);
        }
    }
    cv::Mat core; cv::erode(edges,core,cv::Mat(),cv::Point(-1,-1),1);
    cv::GaussianBlur(core,core,cv::Size(3,3),0.5f);
    for(int y=0;y<f.rows;y++) for(int x=0;x<f.cols;x++) {
        float v=core.at<uchar>(y,x)/255.f*intensity*spark*0.55f;
        auto& p=layer.at<cv::Vec3f>(y,x); p[0]=cf(p[0]+v);p[1]=cf(p[1]+v);p[2]=cf(p[2]+v);
    }
    cv::Mat f32=to32f(f), result(f.rows,f.cols,CV_32FC3);
    for(int y=0;y<f.rows;y++) for(int x=0;x<f.cols;x++) {
        auto b=f32.at<cv::Vec3f>(y,x), g=layer.at<cv::Vec3f>(y,x);
        auto s=blend_screen(b,{cf(g[0]),cf(g[1]),cf(g[2])});
        result.at<cv::Vec3f>(y,x)=s;
    }
    return to8u(result);
}

// --- Fire Outline ---
cv::Mat fire_outline(const cv::Mat& f, float intensity=0.9f, float t=0) {
    cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
    cv::Mat edges; cv::Canny(gray,edges,52,145);
    cv::dilate(edges,edges,cv::Mat(),cv::Point(-1,-1),1);
    float pulse=0.72f+0.28f*(float)std::sin(2*(float)M_PI*3.4f*t+0.6f);
    cv::Mat fire=cv::Mat::zeros(f.rows,f.cols,CV_32FC3);
    float sigs[]={7.f,3.5f,1.8f,0.7f};
    float ir[]={0.88f,0.62f,0.42f,0.85f};
    float ig[]={0.00f,0.20f,0.48f,0.80f};
    float ib[]={0.00f,0.00f,0.10f,0.60f};
    for(int i=0;i<4;i++) {
        cv::Mat bl; cv::GaussianBlur(edges,bl,cv::Size(0,0),sigs[i]);
        for(int y=0;y<f.rows;y++) for(int x=0;x<f.cols;x++) {
            float v=bl.at<uchar>(y,x)/255.f*intensity*pulse;
            auto& p=fire.at<cv::Vec3f>(y,x);
            p[2]=cf(p[2]+v*ir[i]); p[1]=cf(p[1]+v*ig[i]); p[0]=cf(p[0]+v*ib[i]);
        }
    }
    cv::Mat f32=to32f(f), result(f.rows,f.cols,CV_32FC3);
    for(int y=0;y<f.rows;y++) for(int x=0;x<f.cols;x++) {
        auto b=f32.at<cv::Vec3f>(y,x), g=fire.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        r={cf(b[0]+g[0]*0.35f),cf(b[1]+g[1]*0.35f),cf(b[2]+g[2]*0.35f)};
    }
    return to8u(result);
}

// --- Speed Lines ---
cv::Mat speed_lines(const cv::Mat& f, float density=0.85f, float t=0,
                     float cx_f=0.5f, float cy_f=0.5f, int seed=0)
{
    int h=f.rows, w=f.cols;
    std::mt19937 rng(seed^(int)(t*999));
    cv::Mat overlay=cv::Mat::zeros(h,w,CV_8UC3);
    float cx=cx_f*w, cy=cy_f*h, diag=std::sqrt((float)(w*w+h*h));
    int n=(int)(70+density*130);
    std::uniform_real_distribution<float> ang(0,2*(float)M_PI);
    std::uniform_real_distribution<float> ri(0.02f,0.10f), ro(0.30f,0.90f);
    std::uniform_int_distribution<int> lw(1,std::max(1,(int)(density*5)));
    std::uniform_int_distribution<int> al(50,230);
    for(int i=0;i<n;i++) {
        float a=ang(rng), r1=ri(rng)*std::min(w,h), r2=ro(rng)*diag;
        cv::line(overlay,{(int)(cx+r1*std::cos(a)),(int)(cy+r1*std::sin(a))},
                         {(int)(cx+r2*std::cos(a)),(int)(cy+r2*std::sin(a))},
                 cv::Scalar(al(rng),al(rng),al(rng)),lw(rng),cv::LINE_AA);
    }
    cv::GaussianBlur(overlay,overlay,cv::Size(3,3),0.8f);
    cv::Mat f32=to32f(f), ov=to32f(overlay), result(h,w,CV_32FC3);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
        auto b=f32.at<cv::Vec3f>(y,x), g=ov.at<cv::Vec3f>(y,x);
        result.at<cv::Vec3f>(y,x)=blend_screen(b,g);
    }
    return to8u(result);
}

// --- God Rays ---
cv::Mat god_rays(const cv::Mat& f, float intensity=0.85f, float t=0,
                  float cx_f=0.5f, float cy_f=0.06f)
{
    int h=f.rows, w=f.cols;
    cv::Mat f32=to32f(f), lmap(h,w,CV_32F);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) lmap.at<float>(y,x)=luma32(f32.at<cv::Vec3f>(y,x));
    cv::Mat brt; cv::threshold(lmap,brt,0.55f,1,cv::THRESH_TOZERO);
    float lx=cx_f*w, ly=cy_f*h;
    cv::Mat rays=brt.clone(); float decay=0.93f;
    for(int s=1;s<20;s++) {
        float sc=1.f+s*0.007f;
        cv::Mat M=(cv::Mat_<double>(2,3)<<sc,0,(1-sc)*lx,0,sc,(1-sc)*ly);
        cv::Mat w2; cv::warpAffine(brt,w2,M,cv::Size(w,h),cv::INTER_LINEAR,cv::BORDER_CONSTANT);
        rays+=w2*powf(decay,(float)s);
    }
    rays/=20; cv::normalize(rays,rays,0,1,cv::NORM_MINMAX);
    cv::Mat result(h,w,CV_32FC3);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
        auto b=f32.at<cv::Vec3f>(y,x); float r=rays.at<float>(y,x)*intensity*0.52f;
        auto& out=result.at<cv::Vec3f>(y,x);
        out={cf(b[0]+r*0.78f),cf(b[1]+r*0.88f),cf(b[2]+r)};
    }
    return to8u(result);
}

// --- Anamorphic Flare ---
cv::Mat anamorphic_flare(const cv::Mat& f, float intensity=0.80f, float t=0) {
    int h=f.rows, w=f.cols;
    cv::Mat f32=to32f(f), lmap(h,w,CV_32F);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) lmap.at<float>(y,x)=luma32(f32.at<cv::Vec3f>(y,x));
    cv::Mat brt; cv::threshold(lmap,brt,0.78f,1,cv::THRESH_TOZERO);
    cv::Mat streak_src(1,w,CV_32FC3,cv::Scalar(0,0,0));
    for(int x=0;x<w;x++) { float mx=0; for(int y=0;y<h;y++) mx=std::max(mx,brt.at<float>(y,x));
        streak_src.at<cv::Vec3f>(0,x)={mx*0.52f,mx*0.68f,mx}; }
    cv::Mat streak; cv::resize(streak_src,streak,cv::Size(w,h),0,0,cv::INTER_LINEAR);
    cv::GaussianBlur(streak,streak,cv::Size(w|1,3),w*0.045f,2.f);
    float pulse=0.78f+0.22f*(float)std::sin(2*(float)M_PI*0.75f*t+0.3f);
    float blend=(0.12f+0.14f*intensity)*pulse;
    cv::Mat result(h,w,CV_32FC3);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
        auto b=f32.at<cv::Vec3f>(y,x), s=streak.at<cv::Vec3f>(y,x);
        result.at<cv::Vec3f>(y,x)=blend_screen(b,{cf(s[0]*blend),cf(s[1]*blend),cf(s[2]*blend)});
    }
    return to8u(result);
}

// --- Halation (film red bleed in highlights) ---
cv::Mat halation(const cv::Mat& f, float intensity=0.60f) {
    cv::Mat f32=to32f(f);
    std::vector<cv::Mat> ch; cv::split(f32,ch);
    cv::Mat rb; cv::GaussianBlur(ch[2],rb,cv::Size(0,0),20);
    cv::Mat result=f32.clone();
    for(int y=0;y<f32.rows;y++) for(int x=0;x<f32.cols;x++) {
        float lm=luma32(f32.at<cv::Vec3f>(y,x));
        float mask=cf((lm-0.58f)/0.42f); mask*=mask;
        float rbv=rb.at<float>(y,x);
        result.at<cv::Vec3f>(y,x)[2]=cf(f32.at<cv::Vec3f>(y,x)[2]+rbv*mask*intensity*0.55f);
        result.at<cv::Vec3f>(y,x)[1]=cf(f32.at<cv::Vec3f>(y,x)[1]-rbv*mask*intensity*0.06f);
    }
    return to8u(result);
}

// --- Datamosh (pixel drift from optical flow) ---
cv::Mat datamosh(const cv::Mat& f, const cv::Mat& prev, float intensity=0.80f,
                  int idx=0, int seed=0)
{
    if(prev.empty()) return f;
    cv::Mat gc,gp; cv::cvtColor(f,gc,cv::COLOR_BGR2GRAY); cv::cvtColor(prev,gp,cv::COLOR_BGR2GRAY);
    cv::Mat flow;
    try{ cv::calcOpticalFlowFarneback(gp,gc,flow,0.5,2,8,2,5,1.1,0); }
    catch(...){ return f; }
    std::vector<cv::Mat> fch; cv::split(flow,fch);
    int h=f.rows,w=f.cols;
    float sc=intensity*(5.f+std::sin(idx*0.3f)*2.f);
    std::mt19937 rng(seed^idx);
    std::uniform_real_distribution<float> noise(-1.5f,1.5f);
    cv::Mat mx(h,w,CV_32F),my(h,w,CV_32F);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
        mx.at<float>(y,x)=clamp((float)x+fch[0].at<float>(y,x)*sc+noise(rng),0.f,(float)(w-1));
        my.at<float>(y,x)=clamp((float)y+fch[1].at<float>(y,x)*sc+noise(rng),0.f,(float)(h-1));
    }
    cv::Mat dist; cv::remap(prev,dist,mx,my,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    float blend=(idx%8<4)?intensity*0.88f:intensity*0.32f;
    cv::Mat result; cv::addWeighted(f,1-blend,dist,blend,0,result);
    return result;
}

// --- VHS Glitch ---
cv::Mat vhs(const cv::Mat& f, float intensity=0.85f, float t=0, int seed=0) {
    cv::Mat result=f.clone(); int h=f.rows,w=f.cols;
    std::mt19937 rng(seed^(int)(t*7777));
    std::uniform_real_distribution<float> ud(0,1);
    std::uniform_int_distribution<int> bd(0,(int)(intensity*100));
    int nb=2+(int)(intensity*7);
    for(int i=0;i<nb;i++) {
        int y=(int)(ud(rng)*h), bh=2+(int)(ud(rng)*h*0.05f);
        for(int dy=0;dy<bh&&y+dy<h;dy++) for(int x=0;x<w;x++) {
            auto& p=result.at<cv::Vec3b>(y+dy,x); int n=bd(rng);
            p={(uchar)clamp((int)p[0]+n,0,255),(uchar)clamp((int)p[1]+n,0,255),(uchar)clamp((int)p[2]+n,0,255)};
        }
    }
    for(int y=0;y<h;y+=4) for(int x=0;x<w;x++) {
        auto& p=result.at<cv::Vec3b>(y,x); float d=1-0.22f*intensity;
        p={(uchar)(p[0]*d),(uchar)(p[1]*d),(uchar)(p[2]*d)};
    }
    int shift=(int)(intensity*9); if(shift>0&&shift<w) {
        std::vector<cv::Mat> ch; cv::split(result,ch);
        cv::Mat r_sh=cv::Mat::zeros(h,w,CV_8U);
        ch[2](cv::Rect(0,0,w-shift,h)).copyTo(r_sh(cv::Rect(shift,0,w-shift,h)));
        ch[2]=r_sh; cv::merge(ch,result);
    }
    int tl=(int)(ud(rng)*h),th=(int)(ud(rng)*10)+2,ts=(int)((ud(rng)-0.5f)*intensity*38);
    for(int dy=0;dy<th&&tl+dy<h;dy++) {
        cv::Mat row=result.row(tl+dy).clone();
        for(int x=0;x<w;x++) result.at<cv::Vec3b>(tl+dy,x)=row.at<cv::Vec3b>(0,(x+ts+w)%w);
    }
    return result;
}

// --- Flash Frame (curve-driven intensity) ---
cv::Mat flash_frame(const cv::Mat& f, float intensity=0.90f,
                     cv::Vec3f color={1,1,1})
{
    cv::Mat f32=to32f(f), result(f.rows,f.cols,CV_32FC3);
    for(int y=0;y<f32.rows;y++) for(int x=0;x<f32.cols;x++) {
        auto& b=f32.at<cv::Vec3f>(y,x); auto& r=result.at<cv::Vec3f>(y,x);
        r={cf(b[0]*(1-intensity)+color[0]*intensity),
           cf(b[1]*(1-intensity)+color[1]*intensity),
           cf(b[2]*(1-intensity)+color[2]*intensity)};
    }
    return to8u(result);
}

// --- Zoom (bezier-curve driven) ---
cv::Mat zoom_curve(const cv::Mat& f, float t_anim, float dur,
                    float zoom_start=1.0f, float zoom_end=1.15f,
                    const std::string& curve_type="ease")
{
    float t=std::min(1.f,t_anim/std::max(0.01f,dur));
    float ct = (curve_type=="ease_in")  ? Curve::ease_in(t)  :
               (curve_type=="ease_out") ? Curve::ease_out(t) :
               (curve_type=="spring")   ? Curve::spring(t)   :
               (curve_type=="bounce")   ? Curve::bounce_out(t): Curve::ease(t);
    float sc=lerp(zoom_start,zoom_end,ct);
    float cx=f.cols/2.f, cy=f.rows/2.f;
    cv::Mat M=(cv::Mat_<double>(2,3)<<sc,0,(1-sc)*cx,0,sc,(1-sc)*cy);
    cv::Mat r; cv::warpAffine(f,r,M,f.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    return r;
}

// --- Radial Blur (zoom blur) ---
cv::Mat radial_blur(const cv::Mat& f, float amount=0.07f,
                     float cx_f=0.5f, float cy_f=0.5f)
{
    int h=f.rows,w=f.cols; float cx=cx_f*w, cy=cy_f*h;
    int steps=std::max(5,(int)(amount*22));
    cv::Mat f32=to32f(f), acc=cv::Mat::zeros(h,w,CV_32FC3);
    for(int s=0;s<=steps;s++) {
        float t2=(float)s/steps, sc=1.f+t2*amount*0.45f;
        cv::Mat M=(cv::Mat_<double>(2,3)<<sc,0,(1-sc)*cx,0,sc,(1-sc)*cy);
        cv::Mat w2; cv::warpAffine(f32,w2,M,cv::Size(w,h),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        acc+=w2;
    }
    return to8u(acc/(steps+1));
}

// --- Turbulent Displace ---
cv::Mat turbulent_displace(const cv::Mat& f, float amount=28, float evo=0, int seed=42) {
    int h=f.rows,w=f.cols;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> od(0,1000);
    float ox=od(rng), oy=od(rng), freq=1.f/90.f;
    cv::Mat mx(h,w,CV_32F),my(h,w,CV_32F);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
        float nx=0,ny=0,amp=1,f2=freq;
        for(int oct=0;oct<4;oct++){
            float px=(x+ox)*f2, py=(y+oy)*f2;
            nx+=amp*(std::sin(px*1.3f+evo)*std::cos(py*0.7f-evo*0.6f));
            ny+=amp*(std::cos(px*0.9f-evo*0.8f)*std::sin(py*1.1f+evo*0.5f));
            amp*=0.5f; f2*=2.f;
        }
        mx.at<float>(y,x)=clamp((float)x+nx*amount,0.f,(float)(w-1));
        my.at<float>(y,x)=clamp((float)y+ny*amount,0.f,(float)(h-1));
    }
    cv::Mat r; cv::remap(f,r,mx,my,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// --- Contrast + Saturation ---
cv::Mat contrast_sat(const cv::Mat& f, float contrast=0.45f, float saturation=0.40f) {
    cv::Mat f32=to32f(f);
    for(int y=0;y<f32.rows;y++) for(int x=0;x<f32.cols;x++) {
        auto& p=f32.at<cv::Vec3f>(y,x);
        float lm=luma32(p);
        for(int c=0;c<3;c++) {
            p[c]=cf(0.5f+(p[c]-0.5f)*(1.f+contrast));
            p[c]=cf(lm+(p[c]-lm)*(1.f+saturation));
        }
    }
    return to8u(f32);
}

// =============================================================================
// IMPACT FRAME — Manga-style hit frame
// =============================================================================
cv::Mat impact_frame(const cv::Mat& f, float intensity=1.0f, float t=0,
                      bool lines=true, bool flash=true)
{
    cv::Mat result=f.clone();
    // 1. Desaturate + high contrast
    cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
    cv::Mat gray3; cv::cvtColor(gray,gray3,cv::COLOR_GRAY2BGR);
    float desat=intensity*0.85f;
    cv::addWeighted(result,1-desat,gray3,desat,0,result);
    // 2. Boost contrast hard
    result.convertTo(result,-1,1+intensity*1.2f,-intensity*30);
    // 3. Speed lines
    if(lines) result=speed_lines(result,intensity*0.90f,t,0.5f,0.5f,42);
    // 4. Radial blur
    result=radial_blur(result,intensity*0.10f);
    // 5. Chromatic spike
    result=chromatic(result,intensity*0.95f,t);
    // 6. White border flash
    if(flash) {
        int border=(int)(intensity*8);
        cv::rectangle(result,{0,0},{result.cols-1,result.rows-1},
                      cv::Scalar(255,255,255),border);
    }
    return result;
}

} // namespace FX

// =============================================================================
// 8. TRANSITIONS ENGINE — Curve-driven timing
// =============================================================================

namespace Trans {

cv::Mat blend(const cv::Mat& a, const cv::Mat& b, float t) {
    cv::Mat a32=to32f(a), b32=to32f(b), r(a.rows,a.cols,CV_32FC3);
    for(int y=0;y<a.rows;y++) for(int x=0;x<a.cols;x++) {
        auto& pa=a32.at<cv::Vec3f>(y,x); auto& pb=b32.at<cv::Vec3f>(y,x);
        auto& pr=r.at<cv::Vec3f>(y,x);
        pr={lerp(pa[0],pb[0],t),lerp(pa[1],pb[1],t),lerp(pa[2],pb[2],t)};
    }
    return to8u(r);
}

// All transitions use CURVE-DRIVEN t — not linear
cv::Mat apply(const std::string& type, const cv::Mat& a, const cv::Mat& b, float t_linear) {
    cv::Mat b_r=b;
    if(b.size()!=a.size()) cv::resize(b,b_r,a.size());
    t_linear=cf(t_linear);

    // Curve the timing
    float t_ease = Curve::ease(t_linear);
    float t_in   = Curve::ease_in(t_linear);
    float t_out  = Curve::ease_out(t_linear);

    if(type=="cut")           return t_linear<0.5f ? a : b_r;

    if(type=="dissolve")      return blend(a,b_r,t_ease);

    if(type=="flash_cut") {
        float flash=std::sin(t_linear*(float)M_PI);
        cv::Mat mid=blend(a,b_r,t_ease);
        return FX::flash_frame(mid, flash*0.95f);
    }

    if(type=="zoom_punch") {
        float sa=1.f+t_in*0.40f, sb=1.40f-t_out*0.40f;
        cv::Mat za=FX::zoom_curve(a,t_linear,1,1,sa,"ease_in");
        cv::Mat zb=FX::zoom_curve(b_r,t_linear,1,sb,1,"ease_out");
        return blend(za,zb,t_ease);
    }

    if(type=="whip_pan_right") {
        int len=(int)(std::sin(t_linear*(float)M_PI)*70);
        cv::Mat mid=blend(a,b_r,t_ease);
        if(len>2) { cv::Mat k=cv::Mat::zeros(1,len,CV_32F); k=cv::Scalar(1.f/len);
                    cv::Mat r; cv::filter2D(mid,r,-1,k); return r; }
        return mid;
    }
    if(type=="whip_pan_up") {
        int len=(int)(std::sin(t_linear*(float)M_PI)*65);
        cv::Mat mid=blend(a,b_r,t_ease);
        if(len>2) { cv::Mat k=cv::Mat::zeros(len,1,CV_32F); k=cv::Scalar(1.f/len);
                    cv::Mat r; cv::filter2D(mid,r,-1,k); return r; }
        return mid;
    }

    if(type=="glitch_cut") {
        std::mt19937 rng((int)(t_linear*9999));
        cv::Mat mid=blend(a,b_r,t_linear);
        int h=mid.rows,w=mid.cols;
        std::uniform_int_distribution<int> yd(0,h-1),hd(2,(int)(h*0.07f)+2),shd(-(int)(w*0.08f),(int)(w*0.08f));
        for(int i=0;i<10;i++){int y=yd(rng),bh=hd(rng),sh=shd(rng);if(y+bh>=h)continue;
            cv::Mat row=mid.rowRange(y,y+bh).clone();
            for(int dy=0;dy<bh;dy++) for(int x=0;x<w;x++) mid.at<cv::Vec3b>(y+dy,x)=row.at<cv::Vec3b>(dy,(x+sh+w)%w);}
        return FX::chromatic(mid,0.80f,t_linear);
    }

    if(type=="smear_cut") {
        int len=(int)(std::sin(t_linear*(float)M_PI)*55);
        cv::Mat mid=blend(a,b_r,t_ease);
        if(len>3){ cv::Mat k=cv::Mat::zeros(1,len,CV_32F); k=cv::Scalar(1.f/len);
                   cv::Mat r; cv::filter2D(mid,r,-1,k); return r; }
        return mid;
    }

    if(type=="fade_black") {
        cv::Mat black=cv::Mat::zeros(a.size(),a.type());
        if(t_linear<0.5f) return blend(a,black,Curve::ease_in(t_linear*2));
        return blend(black,b_r,Curve::ease_out((t_linear-0.5f)*2));
    }

    if(type=="fade_in") {
        cv::Mat black=cv::Mat::zeros(b_r.size(),b_r.type());
        return blend(black,b_r,t_out);
    }

    if(type=="cross_zoom") {
        cv::Mat za=FX::radial_blur(a,t_in*0.10f);
        cv::Mat zb=FX::radial_blur(b_r,(1-t_out)*0.10f);
        cv::Mat zza=FX::zoom_curve(za,t_linear,1,1,1.30f,"ease_in");
        return blend(zza,zb,t_ease);
    }

    if(type=="wipe_right") {
        cv::Mat r=a.clone(); int split=(int)(t_ease*a.cols);
        if(split>0&&split<=a.cols) b_r(cv::Rect(0,0,split,b_r.rows)).copyTo(r(cv::Rect(0,0,split,r.rows)));
        return r;
    }
    if(type=="wipe_up") {
        cv::Mat r=a.clone(); int split=(int)((1-t_ease)*a.rows);
        if(split<a.rows) b_r(cv::Rect(0,split,b_r.cols,b_r.rows-split)).copyTo(r(cv::Rect(0,split,r.cols,r.rows-split)));
        return r;
    }
    if(type=="energy_slash") {
        int h=a.rows,w=a.cols; cv::Mat r=a.clone();
        float edge_x=t_ease*(w+h);
        for(int y=0;y<h;y++) for(int x=0;x<w;x++){
            float diag=(float)(x+y);
            if(diag<edge_x-25) r.at<cv::Vec3b>(y,x)=b_r.at<cv::Vec3b>(y,x);
            else if(diag<edge_x){ float bt=(diag-(edge_x-25))/25.f;
                auto pa=a.at<cv::Vec3b>(y,x),pb=b_r.at<cv::Vec3b>(y,x);
                r.at<cv::Vec3b>(y,x)={(uchar)lerp(pa[0],pb[0],bt),(uchar)lerp(pa[1],pb[1],bt),(uchar)lerp(pa[2],pb[2],bt)};}
        }
        return r;
    }
    if(type=="luma_wipe") {
        cv::Mat r=a.clone();
        for(int y=0;y<a.rows;y++) for(int x=0;x<a.cols;x++) {
            float lm=luma8(a.at<cv::Vec3b>(y,x));
            if(lm<t_ease) r.at<cv::Vec3b>(y,x)=b_r.at<cv::Vec3b>(y,x);
        }
        return r;
    }
    if(type=="ripple") {
        int h=a.rows,w=a.cols;
        cv::Mat mx(h,w,CV_32F),my(h,w,CV_32F);
        float amp=std::sin(t_linear*(float)M_PI)*18;
        for(int y=0;y<h;y++) for(int x=0;x<w;x++){
            float d=std::sqrt((float)((x-w/2)*(x-w/2)+(y-h/2)*(y-h/2)));
            float disp=amp*(float)std::sin(d*0.08f-t_linear*10);
            mx.at<float>(y,x)=clamp((float)x+disp,0.f,(float)(w-1));
            my.at<float>(y,x)=clamp((float)y+disp*0.5f,0.f,(float)(h-1));
        }
        cv::Mat dist; cv::remap(blend(a,b_r,t_ease),dist,mx,my,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
        return dist;
    }
    if(type=="spin") {
        float angle=t_ease*360.f, sc=1.f-std::sin(t_linear*(float)M_PI)*0.15f;
        cv::Mat mid=blend(a,b_r,t_ease);
        cv::Mat M=cv::getRotationMatrix2D({mid.cols/2.f,mid.rows/2.f},angle,sc);
        cv::Mat r; cv::warpAffine(mid,r,M,mid.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        return r;
    }
    if(type=="shatter_cut") {
        if(t_linear<0.5f){
            float prog=Curve::ease_in(t_linear*2);
            std::mt19937 rng(42); cv::Mat result=a.clone();
            int h=a.rows,w=a.cols;
            std::uniform_real_distribution<float> ud(0,1);
            for(int i=0;i<55;i++){
                int x=(int)(ud(rng)*w),y=(int)(ud(rng)*h);
                int pw=18+(int)(ud(rng)*65),ph=18+(int)(ud(rng)*65);
                int x2=std::min(x+pw,w),y2=std::min(y+ph,h);
                float vx=(ud(rng)-0.5f)*prog*55, vy=(ud(rng)-0.5f)*prog*55;
                cv::Rect roi(x,y,x2-x,y2-y);
                cv::Mat piece=a(roi).clone();
                cv::Mat M2=(cv::Mat_<double>(2,3)<<1,0,vx,0,1,vy);
                cv::warpAffine(piece,result(roi),M2,roi.size(),cv::INTER_LINEAR,cv::BORDER_TRANSPARENT);
            }
            return result;
        }
        return b_r;
    }
    if(type=="tv_static") {
        std::mt19937 rng((int)(t_linear*7777));
        cv::Mat mid=blend(a,b_r,t_ease);
        float static_t=std::sin(t_linear*(float)M_PI);
        if(static_t>0.1f){
            std::uniform_int_distribution<int> nd(0,255);
            for(int y=0;y<mid.rows;y++) for(int x=0;x<mid.cols;x++){
                int v=nd(rng); auto& p=mid.at<cv::Vec3b>(y,x);
                p[0]=(uchar)(p[0]*(1-static_t)+v*static_t);
                p[1]=(uchar)(p[1]*(1-static_t)+v*static_t);
                p[2]=(uchar)(p[2]*(1-static_t)+v*static_t);
            }
        }
        return mid;
    }
    if(type=="dolly_zoom_trans") {
        float zoom_in=t_ease, pull=1-t_ease;
        cv::Mat za=FX::dolly_zoom(a,0,zoom_in*0.3f);
        cv::Mat zb=FX::dolly_zoom(b_r,pull*0.3f,0);
        return blend(za,zb,t_ease);
    }
    if(type=="impact_cut") {
        cv::Mat mid=blend(a,b_r,t_linear);
        float fi=std::sin(t_linear*(float)M_PI);
        if(fi>0.2f) return FX::impact_frame(mid,fi*0.85f,t_linear,true,true);
        return mid;
    }
    // Default
    return t_linear<0.5f ? a : b_r;
}

} // namespace Trans

// =============================================================================
// 9. VIDEO I/O AND PIPELINE
// =============================================================================

static bool RUN_FF(const std::string& args) {
    std::string cmd="ffmpeg -hide_banner -loglevel error -y "+args;
    return std::system(cmd.c_str())==0;
}

static bool source_has_audio(const std::string& path) {
    if(path.empty()) return false;
    fs::path tmp = fs::temp_directory_path()/("arch_audio_probe_"+std::to_string(std::rand())+".txt");
    std::string cmd =
        "ffprobe -v error -select_streams a:0 -show_entries stream=index "
        "-of csv=p=0 \"" + path + "\" > \"" + tmp.string() + "\" 2>nul";
    int rc = std::system(cmd.c_str());
    if(rc != 0) {
        fs::remove(tmp);
        return false;
    }
    std::ifstream in(tmp);
    std::string line;
    std::getline(in, line);
    fs::remove(tmp);
    return !line.empty();
}

static bool encode_frames(const fs::path& dir, const std::string& out,
    double fps, const std::string& audio="", int crf=16,
    const std::string& preset="veryfast")
{
    std::ostringstream c;
    c<<" -framerate "<<fps<<" -i \""<<(dir/"f_%06d.png").string()<<"\"";
    if(source_has_audio(audio)) c<<" -i \""<<audio<<"\" -map 0:v -map 1:a -c:a aac -b:a 192k -shortest";
    c<<" -c:v libx264 -preset "<<preset<<" -crf "<<crf<<" -pix_fmt yuv420p -movflags +faststart \""<<out<<"\"";
    return RUN_FF(c.str());
}

struct VInfo { int w=1280,h=720,total=0; double fps=30,dur=0; };
static VInfo probe(const std::string& p) {
    VInfo v; cv::VideoCapture cap(p);
    if(!cap.isOpened()) return v;
    v.w=(int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    v.h=(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    v.fps=cap.get(cv::CAP_PROP_FPS); if(v.fps<=0)v.fps=30;
    v.total=(int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    v.dur=v.total/v.fps; cap.release(); return v;
}

// =============================================================================
// 10. COMMANDS — Called from Python via subprocess
// =============================================================================

// --- CMD: render_clip ---
// Apply full effect stack to a clip, write to output
// Args: input output mood intensity effect anim_name anim_intensity
//       beat_times(csv) bpm is_drop crf preset w h fps
static bool cmd_render_clip(int argc, char* argv[]) {
    if(argc<5){std::cerr<<"render_clip needs args\n";return false;}
    std::string input=argv[2], output=argv[3];
    std::string mood=argc>4?argv[4]:"hype";
    float intensity=argc>5?std::stof(argv[5]):0.75f;
    std::string effect=argc>6?argv[6]:"cinematic_grade";
    std::string anim=argc>7?argv[7]:"subtle_drift";
    float anim_int=argc>8?std::stof(argv[8]):0.5f;
    // beat_times as csv
    std::string bt_csv=argc>9?argv[9]:"";
    float bpm=argc>10?std::stof(argv[10]):130.f;
    bool is_drop=argc>11&&std::string(argv[11])=="1";
    int crf=argc>12?std::stoi(argv[12]):16;
    std::string preset=argc>13?argv[13]:"veryfast";
    int tw=argc>14?std::stoi(argv[14]):1280;
    int th=argc>15?std::stoi(argv[15]):720;
    float fps=argc>16?std::stof(argv[16]):30.f;
    // Parse beat times
    BeatCurve bc; bc.bpm=bpm;
    {std::stringstream ss(bt_csv); std::string tok;
     while(std::getline(ss,tok,',')) if(!tok.empty()) bc.beat_times.push_back(std::stof(tok));}

    VInfo vi=probe(input);
    cv::VideoCapture cap(input); if(!cap.isOpened()) return false;

    fs::path tmp_dir=fs::temp_directory_path()/("arch_clip_"+std::to_string(std::rand()));
    fs::create_directories(tmp_dir);

    // Build particle system from effect name
    std::string particle_type="";
    if(effect.find("fire")!=std::string::npos) particle_type="embers";
    else if(effect.find("energy")!=std::string::npos||effect.find("aura")!=std::string::npos) particle_type="sparks";
    else if(mood=="sad"||mood=="emotional") particle_type="petals";
    ParticleSystem* psys=nullptr;
    if(!particle_type.empty()) psys=new ParticleSystem(particle_type, 42);

    cv::Mat prev_frame, prev_gray;
    int fidx=0;
    float clip_dur=(float)vi.dur;

    int target_frames=std::max(1, vi.total);
    while(fidx<target_frames) {
        cv::Mat f; if(!cap.read(f)||f.empty()) break;
        if(f.cols!=tw||f.rows!=th) cv::resize(f,f,cv::Size(tw,th));
        float t_local=(float)fidx/fps;
        float t_norm=cf(t_local/std::max(0.1f,clip_dur));
        float beat_energy=bc.evaluate(t_local);
        float t_since_beat=0.5f;
        if(!bc.beat_times.empty()){
            for(float bt:bc.beat_times){if(t_local>=bt){t_since_beat=t_local-bt;}}
        }

        // === LAYER COMPOSITOR ===
        Compositor comp; comp.set_base(f);

        // 1. Lightweight color grade. Avoid stack-heavy LUT allocation in hot CLI paths.
        f=FX::contrast_sat(f,intensity*0.25f,intensity*0.20f);
        comp.set_base(f);

        // 2. Primary effect (with beat-curve intensity)
        float eff_i=intensity*(0.65f+beat_energy*0.35f);
        cv::Mat effected=f.clone();

        if(effect=="impact_flash_combo"||effect=="zoom_burst_shake") {
            effected=FX::glow(f,0.52f,0.85f,eff_i*1.30f,t_local);
            effected=FX::chromatic(effected,eff_i*0.90f,t_local);
            effected=FX::camera_shake(effected,eff_i*0.70f,t_local,t_since_beat,fidx);
        } else if(effect=="energy_aura"||effect=="dark_surge") {
            cv::Vec3b acolor=(mood=="dark")?cv::Vec3b{200,40,200}:(mood=="sad")?cv::Vec3b{200,160,80}:cv::Vec3b{80,160,255};
            effected=FX::energy_aura(f,eff_i*0.95f,t_local,acolor);
            effected=FX::chromatic(effected,eff_i*0.75f,t_local);
        } else if(effect=="fire_outline"||effect=="god_of_war") {
            effected=FX::fire_outline(f,eff_i*0.90f,t_local);
            effected=FX::god_rays(effected,eff_i*0.65f,t_local);
        } else if(effect=="emotional_peak"||effect=="portrait_enhance") {
            effected=FX::glow(f,0.42f,1.0f,eff_i*1.25f,t_local);
            effected=FX::halation(effected,eff_i*0.70f);
        } else if(effect=="speed_blur_lines"||effect=="action") {
            effected=FX::speed_lines(f,eff_i*0.90f,t_local,0.5f,0.5f,fidx);
            if(!prev_frame.empty()) effected=FX::directional_mb_flow(effected,prev_gray,eff_i*0.8f,10);
        } else if(effect=="atmospheric_grade"||effect=="scenery") {
            effected=FX::god_rays(f,eff_i*0.80f,t_local);
            effected=FX::anamorphic_flare(effected,eff_i*0.60f,t_local);
        } else if(effect=="datamosh_corrupt") {
            if(!prev_frame.empty()) effected=FX::datamosh(f,prev_frame,eff_i*0.85f,fidx,fidx);
        } else if(effect=="vhs_glitch") {
            effected=FX::vhs(f,eff_i*0.90f,t_local,fidx);
        } else if(effect=="impact_frame") {
            effected=FX::impact_frame(f,eff_i,t_local,true,beat_energy>0.5f);
        } else if(effect=="dolly_zoom") {
            effected=FX::dolly_zoom(f,t_norm,1-t_norm);
        } else {
            // cinematic_grade / default
            effected=FX::contrast_sat(f,eff_i*0.40f,eff_i*0.35f);
        }

        // 3. Mask-based: character glow (only on action/portrait scenes)
        if(effect=="portrait_enhance"||effect=="emotional_peak") {
            cv::Mat char_mask=MaskEngine::center_ellipse_mask(f);
            cv::Mat char_glow=FX::glow(effected,0.50f,0.8f,eff_i*0.90f,t_local);
            effected=MaskEngine::apply_masked(effected,char_glow,char_mask,false);
            // BG blur outside mask
            cv::Mat bg_blur; cv::GaussianBlur(effected,bg_blur,cv::Size(0,0),8);
            effected=MaskEngine::apply_masked(bg_blur,effected,char_mask,false);
        }

        // 4. Beat-hit effects (curve-driven, not just if/else)
        if(beat_energy>0.05f) {
            // Vignette pulses with beat
            float vign_i=intensity*0.65f+beat_energy*0.25f;
            effected=FX::vignette(effected,vign_i,t_local);
            // Flash spike on fresh beat
            if(t_since_beat<0.04f) {
                float flash_i=Curve::beat_decay(t_since_beat)*0.35f*intensity;
                effected=FX::flash_frame(effected,flash_i);
            }
        } else {
            effected=FX::vignette(effected,intensity*0.60f,t_local);
        }

        // 5. Drop: extra effects when it's a drop frame
        if(is_drop&&fidx<(int)(fps*0.12f)) {
            float drop_t=1.f-(float)fidx/(fps*0.12f);
            drop_t=Curve::ease_out(drop_t);
            effected=FX::camera_shake(effected,drop_t*0.90f,t_local,0,fidx);
            effected=FX::chromatic(effected,drop_t*0.85f,t_local);
            if(fidx<(int)(fps*0.06f))
                effected=FX::flash_frame(effected,drop_t*0.45f);
        }

        // 6. Particle layer (composite over everything)
        if(psys) {
            float em_energy=beat_energy>0.3f ? beat_energy : 0.f;
            if(em_energy>0||fidx%6==0) psys->emit(tw,th,t_local,em_energy+0.2f);
            psys->update(1.f/fps,tw,th);
            cv::Mat player=psys->render_layer(tw,th);
            cv::Mat player_gray;
            cv::cvtColor(player, player_gray, cv::COLOR_BGR2GRAY);
            if(!player.empty() && cv::countNonZero(player_gray) > 0) {
                comp.set_base(effected);
                comp.add_layer(player,0.75f,BlendMode::Screen);
                effected=comp.composite();
            }
        }

        // 7. Film grain always last
        effected=FX::film_grain(effected,intensity*0.45f,t_local,fidx);

        // 8. Animation (curve-driven camera movement)
        float anim_t=t_local;
        if(anim=="ken_burns_slow") {
            float sc=1.f+Curve::ease_in(t_norm)*0.09f*anim_int;
            float cx=f.cols/2.f, cy=f.rows/2.f;
            cv::Mat M=(cv::Mat_<double>(2,3)<<sc,0,(1-sc)*cx,0,sc,(1-sc)*cy);
            cv::warpAffine(effected,effected,M,effected.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        } else if(anim=="handheld_shake") {
            float amp=anim_int*0.016f;
            float dx=amp*effected.cols*(std::sin(anim_t*7.3f)*0.6f+std::sin(anim_t*13.7f)*0.4f);
            float dy=amp*effected.rows*(std::cos(anim_t*6.1f)*0.6f+std::cos(anim_t*11.3f)*0.4f)*0.7f;
            float rot=anim_int*0.5f*(std::sin(anim_t*5.8f)*0.5f+std::sin(anim_t*9.2f)*0.5f);
            cv::Mat M=cv::getRotationMatrix2D({effected.cols/2.f,effected.rows/2.f},rot,1.0);
            M.at<double>(0,2)+=dx; M.at<double>(1,2)+=dy;
            cv::warpAffine(effected,effected,M,effected.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        } else if(anim=="push_in_slow") {
            float sc=1.f+Curve::ease(t_norm)*0.12f*anim_int;
            float cx=effected.cols/2.f, cy=effected.rows/2.f;
            cv::Mat M=(cv::Mat_<double>(2,3)<<sc,0,(1-sc)*cx,0,sc,(1-sc)*cy);
            cv::warpAffine(effected,effected,M,effected.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        } else if(anim=="camera_slam") {
            float hit=std::max(0.f,1.f-t_norm*10.f);
            hit=Curve::ease_in(hit);
            std::mt19937 rng2(0); std::uniform_real_distribution<float> ud(-1,1);
            float dx=hit*ud(rng2)*0.03f*effected.cols, dy=hit*ud(rng2)*0.02f*effected.rows;
            cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,dx,0,1,dy);
            cv::warpAffine(effected,effected,M,effected.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        } else if(anim=="zoom_pulse") {
            float pulse=0.5f+0.5f*(float)std::sin(2*(float)M_PI*4.f*t_norm);
            float sc=1.f+pulse*0.07f*anim_int;
            float cx=effected.cols/2.f, cy=effected.rows/2.f;
            cv::Mat M=(cv::Mat_<double>(2,3)<<sc,0,(1-sc)*cx,0,sc,(1-sc)*cy);
            cv::warpAffine(effected,effected,M,effected.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        } else if(anim=="parallax_zoom") {
            float sc=1.f+Curve::ease(t_norm)*0.14f*anim_int;
            float ox=std::sin(t_norm*(float)M_PI)*0.04f*anim_int*effected.cols;
            float cx=effected.cols/2.f, cy=effected.rows/2.f;
            cv::Mat M=(cv::Mat_<double>(2,3)<<sc,0,(1-sc)*cx+ox,0,sc,(1-sc)*cy);
            cv::warpAffine(effected,effected,M,effected.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        } else if(anim=="subtle_drift") {
            float dx=std::sin(anim_t*0.4f)*0.012f*anim_int*effected.cols;
            float dy=std::cos(anim_t*0.3f)*0.008f*anim_int*effected.rows;
            float sc=1.f+std::sin(anim_t*0.25f)*0.015f*anim_int;
            float cx=effected.cols/2.f, cy=effected.rows/2.f;
            cv::Mat M=(cv::Mat_<double>(2,3)<<sc,0,(1-sc)*cx+dx,0,sc,(1-sc)*cy+dy);
            cv::warpAffine(effected,effected,M,effected.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        } else if(anim=="dolly_zoom_anim") {
            effected=FX::dolly_zoom(effected,t_norm*anim_int,0);
        } else if(anim=="orbit_right") {
            float dx=t_norm*0.09f*anim_int*effected.cols;
            float sc=1.f+t_norm*0.04f*anim_int;
            float cx=effected.cols/2.f, cy=effected.rows/2.f;
            cv::Mat M=(cv::Mat_<double>(2,3)<<sc,0,(1-sc)*cx+dx,0,sc,(1-sc)*cy);
            cv::warpAffine(effected,effected,M,effected.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        }

        // Save frame
        char buf[32]; std::snprintf(buf,sizeof(buf),"f_%06d.png",fidx);
        cv::imwrite((tmp_dir/buf).string(),effected);
        prev_gray=cv::Mat(); cv::cvtColor(f,prev_gray,cv::COLOR_BGR2GRAY);
        prev_frame=f.clone();
        fidx++;
    }
    cap.release();
    if(psys) delete psys;
    if(!fidx){fs::remove_all(tmp_dir);return false;}
    bool ok=encode_frames(tmp_dir,output,fps,input,crf,preset);
    fs::remove_all(tmp_dir);
    return ok;
}

// --- CMD: render_transition ---
// Args: type clip_a clip_b output dur_frames crf preset w h fps
static bool cmd_render_transition(int argc, char* argv[]) {
    if(argc<6){std::cerr<<"render_transition needs args\n";return false;}
    std::string type=argv[2], clip_a=argv[3], clip_b=argv[4], output=argv[5];
    int n_frames=argc>6?std::stoi(argv[6]):15;
    int crf=argc>7?std::stoi(argv[7]):16;
    std::string preset=argc>8?argv[8]:"veryfast";
    int tw=argc>9?std::stoi(argv[9]):1280, th=argc>10?std::stoi(argv[10]):720;
    float fps=argc>11?std::stof(argv[11]):30.f;

    VInfo va=probe(clip_a);
    cv::VideoCapture ca(clip_a), cb(clip_b);
    ca.set(cv::CAP_PROP_POS_FRAMES,std::max(0,va.total-n_frames-1));
    std::vector<cv::Mat> fa,fb;
    while((int)fa.size()<n_frames){cv::Mat f;if(!ca.read(f)||f.empty())break;
        if(f.cols!=tw||f.rows!=th)cv::resize(f,f,cv::Size(tw,th)); fa.push_back(f.clone());}
    while((int)fb.size()<n_frames){cv::Mat f;if(!cb.read(f)||f.empty())break;
        if(f.cols!=tw||f.rows!=th)cv::resize(f,f,cv::Size(tw,th)); fb.push_back(f.clone());}
    ca.release(); cb.release();

    fs::path tmp_dir=fs::temp_directory_path()/("arch_trans_"+std::to_string(std::rand()));
    fs::create_directories(tmp_dir);
    int n=std::min({(int)fa.size(),(int)fb.size(),n_frames});
    for(int i=0;i<n;i++){
        float t=(float)i/std::max(1,n-1);
        cv::Mat r=Trans::apply(type,fa[i],fb[i],t);
        char buf[32]; std::snprintf(buf,sizeof(buf),"f_%06d.png",i);
        cv::imwrite((tmp_dir/buf).string(),r);
    }
    bool ok=encode_frames(tmp_dir,output,fps,"",crf,preset);
    fs::remove_all(tmp_dir); return ok;
}

// --- CMD: apply_effect_single ---
// Apply a single named effect to a video file
static bool cmd_apply_effect(int argc, char* argv[]) {
    if(argc<5){std::cerr<<"apply_effect needs: effect input output [intensity]\n";return false;}
    std::string effect=argv[2], input=argv[3], output=argv[4];
    float intensity=argc>5?std::stof(argv[5]):0.80f;
    std::string mood=argc>6?argv[6]:"hype";
    int crf=argc>7?std::stoi(argv[7]):16;
    std::string preset=argc>8?argv[8]:"veryfast";

    VInfo vi=probe(input);
    cv::VideoCapture cap(input); if(!cap.isOpened()) return false;
    fs::path tmp_dir=fs::temp_directory_path()/("arch_fx_"+std::to_string(std::rand()));
    fs::create_directories(tmp_dir);

    cv::Mat prev, prev_gray; int fidx=0;
    while(true){
        cv::Mat f; if(!cap.read(f)||f.empty()) break;
        float t=(float)fidx/vi.fps;
        cv::Mat r=FX::contrast_sat(f,intensity*0.25f,intensity*0.20f);

        if(effect=="glow")           r=FX::glow(r,0.52f,0.85f,intensity*1.3f,t);
        else if(effect=="chromatic") r=FX::chromatic(r,intensity,t);
        else if(effect=="energy_aura") r=FX::energy_aura(r,intensity,t,{80,160,255});
        else if(effect=="fire_outline") r=FX::fire_outline(r,intensity,t);
        else if(effect=="speed_lines") r=FX::speed_lines(r,intensity,t,0.5f,0.5f,fidx);
        else if(effect=="god_rays")  r=FX::god_rays(r,intensity,t);
        else if(effect=="vignette")  r=FX::vignette(r,intensity,t);
        else if(effect=="film_grain") r=FX::film_grain(r,intensity,t,fidx);
        else if(effect=="datamosh"&&!prev.empty()) r=FX::datamosh(r,prev,intensity,fidx,fidx);
        else if(effect=="vhs")       r=FX::vhs(r,intensity,t,fidx);
        else if(effect=="flash")     r=FX::flash_frame(r,intensity);
        else if(effect=="radial_blur") r=FX::radial_blur(r,intensity*0.08f);
        else if(effect=="turbulent") r=FX::turbulent_displace(r,intensity*30,t*0.5f,42);
        else if(effect=="halation")  r=FX::halation(r,intensity);
        else if(effect=="anamorphic_flare") r=FX::anamorphic_flare(r,intensity,t);
        else if(effect=="impact_frame") r=FX::impact_frame(r,intensity,t);
        else if(effect=="dolly_zoom") r=FX::dolly_zoom(r,(float)fidx/vi.total,0);
        else if(effect=="directional_mb"&&!prev_gray.empty()) r=FX::directional_mb_flow(r,prev_gray,intensity);
        else if(effect=="contrast_sat") r=FX::contrast_sat(r,intensity*0.5f,intensity*0.4f);
        else { // synth: compose from primitives
            r=FX::contrast_sat(r,intensity*0.35f,intensity*0.30f);
            r=FX::glow(r,0.55f,0.7f,intensity*0.80f,t);
            r=FX::vignette(r,intensity*0.65f,t);
        }
        r=FX::film_grain(r,intensity*0.40f,t,fidx);

        char buf[32]; std::snprintf(buf,sizeof(buf),"f_%06d.png",fidx);
        cv::imwrite((tmp_dir/buf).string(),r);
        prev=f.clone(); prev_gray=cv::Mat(); cv::cvtColor(f,prev_gray,cv::COLOR_BGR2GRAY);
        fidx++;
    }
    cap.release();
    bool ok=encode_frames(tmp_dir,output,vi.fps,input,crf,preset);
    fs::remove_all(tmp_dir); return ok;
}

// --- CMD: analyze_clip ---
// Output JSON with scene analysis
static bool cmd_analyze_clip(int argc, char* argv[]) {
    if(argc<3){std::cerr<<"analyze_clip needs: input\n";return false;}
    std::string input=argv[2];
    int n_samples=argc>3?std::stoi(argv[3]):8;
    VInfo vi=probe(input);
    cv::VideoCapture cap(input); if(!cap.isOpened()){std::cout<<"{}";return false;}
    float step=vi.dur/n_samples;
    float tm=0,tb=0,ts=0,tsat=0; int count=0;
    cv::Mat prev_gray;
    for(int s=0;s<n_samples;s++){
        cap.set(cv::CAP_PROP_POS_MSEC,s*step*1000);
        cv::Mat f; if(!cap.read(f)||f.empty()) continue;
        cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
        if(!prev_gray.empty()&&prev_gray.size()==gray.size()){
            cv::Mat flow; cv::calcOpticalFlowFarneback(prev_gray,gray,flow,0.5,2,8,2,5,1.1,0);
            std::vector<cv::Mat> fch; cv::split(flow,fch);
            cv::Mat mag; cv::magnitude(fch[0],fch[1],mag);
            tm+=cf((float)cv::mean(mag)[0]/18.f);
        }
        prev_gray=gray.clone();
        cv::Mat f32=to32f(f);
        float bright=0;
        for(int y=0;y<f32.rows;y++) for(int x=0;x<f32.cols;x++) bright+=luma32(f32.at<cv::Vec3f>(y,x));
        tb+=bright/(f32.rows*f32.cols);
        cv::Mat lap; cv::Laplacian(gray,lap,CV_64F);
        cv::Scalar mu,sig; cv::meanStdDev(lap,mu,sig);
        ts+=cf((float)(sig[0]*sig[0])/4000.f);
        cv::Mat hsv; cv::cvtColor(f,hsv,cv::COLOR_BGR2HSV);
        tsat+=(float)cv::mean(hsv)[1]/255.f;
        count++;
    }
    cap.release();
    if(!count){std::cout<<"{}";return false;}
    float motion=tm/count, bright=tb/count, sharp=ts/count, sat=tsat/count;
    std::string scene_type = motion>0.60f?"impact":motion>0.35f?"action":
                             bright>0.65f&&motion<0.2f?"scenery":motion<0.12f?"quiet":"reaction";
    float edit_val=motion*0.40f+sharp*0.25f+sat*0.20f+(bright*(1-bright)*4)*0.15f;
    // JSON output
    std::cout<<"{"
             <<"\"motion\":"<<motion<<","
             <<"\"brightness\":"<<bright<<","
             <<"\"sharpness\":"<<sharp<<","
             <<"\"saturation\":"<<sat<<","
             <<"\"scene_type\":\""<<scene_type<<"\","
             <<"\"edit_value\":"<<edit_val<<","
             <<"\"duration\":"<<vi.dur<<","
             <<"\"fps\":"<<vi.fps<<","
             <<"\"width\":"<<vi.w<<","
             <<"\"height\":"<<vi.h
             <<"}";
    return true;
}

// --- List all effects and transitions ---
static void cmd_list() {
    std::cout<<"ARCHIVE FRAME ENGINE v4.0\n\n"
             <<"COMMANDS:\n"
             <<"  render_clip      Apply full effect stack to a clip\n"
             <<"  render_transition Render transition between two clips\n"
             <<"  apply_effect     Apply single effect to video\n"
             <<"  analyze_clip     Output JSON scene analysis\n"
             <<"  list             Show this\n\n"
             <<"EFFECTS:\n"
             <<"  glow chromatic camera_shake vignette film_grain\n"
             <<"  energy_aura fire_outline speed_lines god_rays\n"
             <<"  anamorphic_flare halation datamosh vhs flash radial_blur\n"
             <<"  turbulent directional_mb contrast_sat dolly_zoom\n"
             <<"  impact_frame impact_flash_combo speed_blur_lines\n"
             <<"  emotional_peak portrait_enhance atmospheric_grade\n"
             <<"  datamosh_corrupt vhs_glitch scenery action\n\n"
             <<"TRANSITIONS:\n"
             <<"  cut dissolve flash_cut zoom_punch whip_pan_right whip_pan_up\n"
             <<"  glitch_cut smear_cut fade_black fade_in cross_zoom\n"
             <<"  wipe_right wipe_up energy_slash luma_wipe ripple\n"
             <<"  spin shatter_cut tv_static dolly_zoom_trans impact_cut\n\n"
             <<"ANIMATIONS:\n"
             <<"  ken_burns_slow push_in_slow handheld_shake camera_slam\n"
             <<"  zoom_pulse parallax_zoom subtle_drift dolly_zoom_anim\n"
             <<"  orbit_right\n\n"
             <<"MOODS:\n"
             <<"  hype phonk dark villain sad emotional dramatic cinematic\n"
             <<"  cyberpunk neon warm golden chill lofi bleach teal_orange\n\n"
             <<"CURVES (used internally):\n"
             <<"  ease ease_in ease_out ease_in_out spring bounce_out\n"
             <<"  impact punch beat_decay shake_wave\n";
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char* argv[]) {
    if(argc<2){cmd_list();return 0;}
    std::string cmd=argv[1];
    if(cmd=="list"||cmd=="--list"){cmd_list();return 0;}
    if(cmd=="render_clip")       return cmd_render_clip(argc,argv)?0:1;
    if(cmd=="render_transition") return cmd_render_transition(argc,argv)?0:1;
    if(cmd=="apply_effect")      return cmd_apply_effect(argc,argv)?0:1;
    if(cmd=="analyze_clip")      return cmd_analyze_clip(argc,argv)?0:1;
    std::cerr<<"Unknown command: "<<cmd<<"\n";
    cmd_list(); return 1;
}
