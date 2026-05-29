// =============================================================================
// ARCHIVE ULTIMATE EFFECTS ENGINE  v2.0  -  archive_ultimate_effects.cpp
//
// Every professional effect from After Effects + Premiere Pro,
// re-engineered in C++ with OpenCV for maximum quality and speed.
//
// EFFECT CATEGORIES (matching AE's menu structure):
//   BLUR & SHARPEN   - Gaussian, Directional, Radial, Bilateral, Box,
//                      Smart, Camera Lens, Channel, Compound, Unsharp Mask
//   COLOR CORRECTION - Curves, Levels, Hue/Saturation, Color Balance,
//                      Vibrance, Photo Filter, Tint, Lumetri, 3-Way Color,
//                      Shadows/Highlights, Color Grade LUTs (12 moods)
//   DISTORT          - Turbulent Displace, Wave Warp, Ripple, Bulge,
//                      Twirl, Spherize, Mirror, Polar Coords, Liquify,
//                      Displacement Map, Mesh Warp, Lens Distortion
//   GENERATE         - Advanced Lightning, Lens Flare, Fractal Noise,
//                      Cell Pattern, Gradient Ramp, Beam, Particle Rain,
//                      Particle Snow, Particle Sparks, Radio Waves
//   STYLIZE          - Glow, Find Edges, Cartoon, Emboss, Mosaic,
//                      Posterize, Scatter, Threshold, Brush Strokes,
//                      Roughen Edges, Motion Tile, Strobe Light
//   NOISE & GRAIN    - Film Grain, Add Grain, Dust & Scratches,
//                      Fractal Noise, Turbulent Noise, Median
//   TIME EFFECTS     - Echo Trail, Posterize Time, Time Displacement
//   SIMULATION       - Particle World, Rain, Snow, Shatter, Foam, Caustics
//   PERSPECTIVE      - Drop Shadow, Bevel Edges, Radial Shadow
//   KEYING           - Chroma Key, Luma Key, Difference Matte
//   CHANNEL          - Invert, Shift Channels, Set Matte, Minimax
//   CUSTOM AMV       - Optical Flow Slomo (Twixtor), RSMB Motion Blur,
//                      Deep Glow, Energy Aura, Fire Outline, Color Dodge Aura,
//                      Speed Lines, Ink Outline, Datamosh, VHS, Anamorphic Flare,
//                      God Rays, Chromatic Aberration, Light Leak, Twitch
//
// BUILD:
//   cl /std:c++17 /O2 /EHsc /openmp archive_ultimate_effects.cpp ^
//      /I"C:\vcpkg\installed\x64-windows\include" ^
//      /I"C:\vcpkg\installed\x64-windows\include\opencv4" ^
//      /link /LIBPATH:"C:\vcpkg\installed\x64-windows\lib" opencv_world4.lib ^
//      /out:archive_ultimate_effects.exe
//
// USAGE:
//   archive_ultimate_effects.exe --effect <name> --input in.mp4 --output out.mp4 [options]
//   archive_ultimate_effects.exe --preset  <name> --input in.mp4 --output out.mp4
//   archive_ultimate_effects.exe --list
// =============================================================================

#define _USE_MATH_DEFINES
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
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
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;

// =============================================================================
// GLOBALS & HELPERS
// =============================================================================

static const std::string EXE_VERSION = "2.0";
static const std::string FFMPEG_PATH = "ffmpeg"; // must be in PATH

template<typename T>
static inline T clamp(T v, T lo, T hi){ return v<lo?lo:v>hi?hi:v; }
static inline float clampf(float v){ return clamp(v,0.f,1.f); }

static void LOG(const std::string& m){
    std::cout << "[AUE] " << m << "\n"; std::cout.flush();
}

static bool RUN_FFMPEG(const std::string& args, int timeout=900){
    std::string cmd = FFMPEG_PATH + " -hide_banner -loglevel error " + args;
    LOG("FFmpeg: " + cmd.substr(0,100));
    return std::system(cmd.c_str()) == 0;
}

// ---------------- Video info ----------------
struct VInfo {
    int    w=1280, h=720, total=0;
    double fps=30.0, dur=30.0;
    bool   audio=false;
};

static VInfo probe(const std::string& path){
    VInfo v;
    cv::VideoCapture cap(path);
    if(!cap.isOpened()) return v;
    v.w     = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    v.h     = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    v.fps   = cap.get(cv::CAP_PROP_FPS);
    v.total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    if(v.fps>0) v.dur = v.total/v.fps;
    if(v.fps<=0) v.fps=30;
    cap.release();
    return v;
}

// ---------------- Temp dir ------------------
static fs::path mktmp(const char* pfx="ae_"){
    fs::path p = fs::temp_directory_path()
        / (std::string(pfx)+std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(p);
    return p;
}
static void rmtmp(const fs::path& p){
    try{ fs::remove_all(p); }catch(...){}
}

// ---------------- Encode frames to video -----
static bool encode_frames(
    const fs::path& dir,
    const std::string& out,
    double fps,
    const std::string& audio_src="",
    int crf=16,
    const std::string& preset="veryfast"
){
    std::ostringstream c;
    c << " -y -framerate " << fps
      << " -i \"" << (dir/"f_%06d.png").string() << "\"";
    if(!audio_src.empty())
        c << " -i \"" << audio_src << "\""
          << " -map 0:v -map 1:a? -c:a aac -b:a 192k -shortest";
    c << " -c:v libx264 -preset " << preset
      << " -crf " << crf
      << " -pix_fmt yuv420p -movflags +faststart"
      << " \"" << out << "\"";
    return RUN_FFMPEG(c.str());
}

// ---------------- Frame pipeline helper ------
// Applies a per-frame function to the whole video,
// writing PNG frames to a temp dir then encoding.
using FProc = std::function<cv::Mat(const cv::Mat&, float /*t_sec*/, int /*idx*/)>;

static bool pipeline(
    const std::string& in,
    const std::string& out,
    FProc fn,
    int crf=16,
    const std::string& preset="veryfast"
){
    VInfo vi = probe(in);
    cv::VideoCapture cap(in);
    if(!cap.isOpened()){ LOG("Cannot open: "+in); return false; }
    auto tmp = mktmp("pipe_");
    int idx=0;
    LOG("Processing: "+std::to_string(vi.total)+" frames...");
    while(true){
        cv::Mat f;
        if(!cap.read(f)||f.empty()) break;
        float t = (float)(idx/vi.fps);
        cv::Mat r = fn(f,t,idx);
        char buf[24]; std::snprintf(buf,sizeof(buf),"f_%06d.png",idx);
        cv::imwrite((tmp/buf).string(),r);
        idx++;
        if(idx%100==0) LOG("  Frame "+std::to_string(idx)+"/"+std::to_string(vi.total));
    }
    cap.release();
    if(!idx){ rmtmp(tmp); return false; }
    bool ok = encode_frames(tmp,out,vi.fps,vi.audio?in:"",crf,preset);
    rmtmp(tmp);
    return ok;
}

// ---------------- Math helpers ----------------
static inline float lerp(float a,float b,float t){ return a+(b-a)*t; }
static inline float smoothstep(float x){ x=clampf(x); return x*x*(3-2*x); }
static inline float ease_out(float x,float p=2.f){ return 1.f-powf(1.f-clampf(x),p); }

// ---------------- Per-pixel luma --------------
static inline float luma_bgr(const cv::Vec3f& p){
    return 0.114f*p[0]+0.587f*p[1]+0.299f*p[2];
}
static inline float luma_bgr8(const cv::Vec3b& p){
    return (0.114f*p[0]+0.587f*p[1]+0.299f*p[2])/255.f;
}

// Convert to float [0,1] and back
static cv::Mat to32f(const cv::Mat& m){
    cv::Mat r; m.convertTo(r,CV_32FC3,1.0/255.0); return r;
}
static cv::Mat to8u(const cv::Mat& m){
    cv::Mat r; m.convertTo(r,CV_8UC3,255.0); return r;
}

// Screen blend
static cv::Vec3f screen(const cv::Vec3f& a,const cv::Vec3f& b){
    return {1-(1-a[0])*(1-b[0]),
            1-(1-a[1])*(1-b[1]),
            1-(1-a[2])*(1-b[2])};
}
// Overlay blend
static float overlay1(float a,float b){
    return a<0.5f ? 2*a*b : 1-2*(1-a)*(1-b);
}
// Color dodge
static float dodge1(float b,float g){
    return clampf(b/std::max(1e-4f,1.f-g));
}

// =============================================================================
// SECTION 1 - BLUR & SHARPEN
// =============================================================================

// --- 1.1 Gaussian Blur ---
static cv::Mat fx_gaussian_blur(const cv::Mat& f,float sigma){
    int k=(int)(sigma*3)|1; if(k<3)k=3;
    cv::Mat r; cv::GaussianBlur(f,r,cv::Size(k,k),sigma); return r;
}

// --- 1.2 Directional (Motion) Blur ---
static cv::Mat fx_directional_blur(const cv::Mat& f,int len,float angle_deg){
    cv::Mat kernel=cv::Mat::zeros(len,len,CV_32F);
    float cx=len/2.f, cy=len/2.f;
    float rad=angle_deg*(float)M_PI/180.f;
    float dx=std::cos(rad), dy=std::sin(rad);
    for(int i=0;i<len;i++){
        int x=(int)(cx+dx*(i-len/2)+0.5f);
        int y=(int)(cy+dy*(i-len/2)+0.5f);
        if(x>=0&&x<len&&y>=0&&y<len) kernel.at<float>(y,x)=1.f;
    }
    float s=cv::sum(kernel)[0]; if(s>0) kernel/=s;
    cv::Mat r; cv::filter2D(f,r,-1,kernel); return r;
}

// --- 1.3 Radial Blur (zoom) ---
static cv::Mat fx_radial_blur(const cv::Mat& f,float amount,
                               float cx_f=0.5f,float cy_f=0.5f){
    int h=f.rows,w=f.cols;
    float cx=cx_f*w,cy=cy_f*h;
    int steps=std::max(4,(int)(amount*16));
    cv::Mat acc=cv::Mat::zeros(h,w,CV_32FC3);
    cv::Mat f32=to32f(f);
    for(int s=0;s<=steps;s++){
        float t=(float)s/steps;
        float scale=1.f+t*amount*0.3f;
        cv::Mat M=(cv::Mat_<double>(2,3)<<
            scale,0,(1-scale)*cx,
            0,scale,(1-scale)*cy);
        cv::Mat warped;
        cv::warpAffine(f32,warped,M,cv::Size(w,h),
                       cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        acc+=warped;
    }
    acc/=(steps+1);
    return to8u(acc);
}

// --- 1.4 Bilateral Blur (edge-preserving) ---
static cv::Mat fx_bilateral_blur(const cv::Mat& f,int d,
                                  float sigma_color,float sigma_space){
    cv::Mat r;
    cv::bilateralFilter(f,r,d,sigma_color,sigma_space);
    return r;
}

// --- 1.5 Box Blur ---
static cv::Mat fx_box_blur(const cv::Mat& f,int radius){
    cv::Mat r;
    int k=2*radius+1;
    cv::blur(f,r,cv::Size(k,k));
    return r;
}

// --- 1.6 Camera Lens Blur (bokeh simulation) ---
static cv::Mat fx_camera_lens_blur(const cv::Mat& f,int radius,
                                    float iris_blades=6){
    // Create hexagonal/polygonal bokeh kernel
    int k=2*radius+1;
    cv::Mat kernel=cv::Mat::zeros(k,k,CV_32F);
    float cx=radius, cy=radius;
    std::vector<cv::Point> pts;
    int n=(int)iris_blades; if(n<3)n=6;
    for(int i=0;i<n;i++){
        float a=2*(float)M_PI*i/n - (float)M_PI/2;
        pts.push_back({(int)(cx+radius*std::cos(a)),
                       (int)(cy+radius*std::sin(a))});
    }
    cv::fillConvexPoly(kernel,pts,cv::Scalar(1.f));
    float s=cv::sum(kernel)[0]; if(s>0) kernel/=s;
    cv::Mat r; cv::filter2D(f,r,-1,kernel);
    return r;
}

// --- 1.7 Smart Blur (blur only flat regions) ---
static cv::Mat fx_smart_blur(const cv::Mat& f,float sigma,float threshold){
    cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
    cv::Mat edges; cv::Canny(gray,edges,threshold*0.5,threshold);
    // Dilate edge mask
    cv::Mat emask; cv::dilate(edges,emask,cv::Mat(),cv::Point(-1,-1),2);
    cv::Mat blurred; cv::GaussianBlur(f,blurred,cv::Size(0,0),sigma);
    cv::Mat result=f.clone();
    // Only apply blur where NOT an edge
    for(int y=0;y<f.rows;y++)
    for(int x=0;x<f.cols;x++){
        if(!emask.at<uchar>(y,x)){
            result.at<cv::Vec3b>(y,x)=blurred.at<cv::Vec3b>(y,x);
        }
    }
    return result;
}

// --- 1.8 Unsharp Mask ---
static cv::Mat fx_unsharp_mask(const cv::Mat& f,float amount,float radius){
    cv::Mat blurred; cv::GaussianBlur(f,blurred,cv::Size(0,0),radius);
    cv::Mat result;
    cv::addWeighted(f,1.f+amount,blurred,-amount,0,result);
    return result;
}

// --- 1.9 Channel Blur (per-channel) ---
static cv::Mat fx_channel_blur(const cv::Mat& f,
                                float sigma_r,float sigma_g,float sigma_b){
    std::vector<cv::Mat> ch; cv::split(f,ch);
    if(sigma_b>0) cv::GaussianBlur(ch[0],ch[0],cv::Size(0,0),sigma_b);
    if(sigma_g>0) cv::GaussianBlur(ch[1],ch[1],cv::Size(0,0),sigma_g);
    if(sigma_r>0) cv::GaussianBlur(ch[2],ch[2],cv::Size(0,0),sigma_r);
    cv::Mat r; cv::merge(ch,r); return r;
}

// --- 1.10 Compound Blur (blur controlled by luminance map) ---
static cv::Mat fx_compound_blur(const cv::Mat& f,float max_blur){
    cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
    cv::Mat result=f.clone();
    // Apply different blur levels based on brightness
    for(int level=0;level<5;level++){
        float t=(float)level/4;
        float sigma=max_blur*t;
        if(sigma<0.5f) continue;
        cv::Mat blurred; cv::GaussianBlur(f,blurred,cv::Size(0,0),sigma);
        cv::Mat mask=(gray<=(int)(55+t*200))&(gray>=(int)(t*200-55));
        blurred.copyTo(result,mask);
    }
    return result;
}

// =============================================================================
// SECTION 2 - COLOR CORRECTION
// =============================================================================

// --- 2.1 Curves (per-channel S-curve or custom) ---
struct CurveParams {
    // Control points: each is (input,output) in [0,1]
    std::vector<std::pair<float,float>> r={{0.f,0.f},{1.f,1.f}};
    std::vector<std::pair<float,float>> g={{0.f,0.f},{1.f,1.f}};
    std::vector<std::pair<float,float>> b={{0.f,0.f},{1.f,1.f}};
    std::vector<std::pair<float,float>> master={{0.f,0.f},{1.f,1.f}};
};

static float curve_interp(const std::vector<std::pair<float,float>>& pts,float x){
    if(pts.size()<2) return x;
    x=clampf(x);
    // Simple linear interpolation between control points
    for(int i=0;i<(int)pts.size()-1;i++){
        float x0=pts[i].first, y0=pts[i].second;
        float x1=pts[i+1].first, y1=pts[i+1].second;
        if(x>=x0&&x<=x1){
            float t=(x-x0)/(x1-x0);
            // Hermite interpolation for smooth curves
            t=smoothstep(t);
            return clampf(y0+(y1-y0)*t);
        }
    }
    return x;
}

static cv::Mat fx_curves(const cv::Mat& f,const CurveParams& cp){
    cv::Mat f32=to32f(f);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        p[0]=curve_interp(cp.b,curve_interp(cp.master,p[0]));
        p[1]=curve_interp(cp.g,curve_interp(cp.master,p[1]));
        p[2]=curve_interp(cp.r,curve_interp(cp.master,p[2]));
    }
    return to8u(f32);
}

// --- 2.2 Levels ---
static cv::Mat fx_levels(const cv::Mat& f,
    float in_black=0,float in_white=1,float gamma=1,
    float out_black=0,float out_white=1){
    cv::Mat f32=to32f(f);
    float range=std::max(1e-5f,in_white-in_black);
    float out_range=out_white-out_black;
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        for(int c=0;c<3;c++){
            float v=clampf((p[c]-in_black)/range);
            if(gamma!=1.f&&v>0) v=powf(v,1.f/gamma);
            p[c]=clampf(out_black+v*out_range);
        }
    }
    return to8u(f32);
}

// --- 2.3 Hue/Saturation/Lightness ---
static cv::Mat fx_hue_sat_light(const cv::Mat& f,
    float hue_shift,float sat_mult,float light_add){
    cv::Mat hsv; cv::cvtColor(f,hsv,cv::COLOR_BGR2HSV);
    for(int y=0;y<hsv.rows;y++)
    for(int x=0;x<hsv.cols;x++){
        auto& p=hsv.at<cv::Vec3b>(y,x);
        // Hue: [0,180] in OpenCV
        p[0]=(uchar)(((int)p[0]+(int)(hue_shift*0.5f)+180)%180);
        // Sat: [0,255]
        p[1]=(uchar)clamp((int)(p[1]*sat_mult),0,255);
        // Val: [0,255]
        p[2]=(uchar)clamp((int)(p[2]+light_add*255.f),0,255);
    }
    cv::Mat r; cv::cvtColor(hsv,r,cv::COLOR_HSV2BGR);
    return r;
}

// --- 2.4 Color Balance (shadows/midtones/highlights) ---
static cv::Mat fx_color_balance(const cv::Mat& f,
    float sr,float sg,float sb,  // shadow R/G/B shift
    float mr,float mg,float mb,  // midtone
    float hr,float hg,float hb){ // highlight
    cv::Mat f32=to32f(f);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        float lm=luma_bgr(p);
        float sw=powf(clampf(1.f-lm*2.f),1.5f);  // shadow weight
        float hw=powf(clampf(lm*2.f-1.f),1.5f);   // highlight weight
        float mw=1.f-sw-hw;                         // midtone weight
        p[2]=clampf(p[2]+sr*sw+mr*mw+hr*hw);
        p[1]=clampf(p[1]+sg*sw+mg*mw+hg*hw);
        p[0]=clampf(p[0]+sb*sw+mb*mw+hb*hw);
    }
    return to8u(f32);
}

// --- 2.5 Vibrance (smart saturation: less effect on already-saturated) ---
static cv::Mat fx_vibrance(const cv::Mat& f,float vibrance,float saturation){
    cv::Mat f32=to32f(f);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        float lm=luma_bgr(p);
        // Current saturation
        float mx=std::max({p[0],p[1],p[2]});
        float mn=std::min({p[0],p[1],p[2]});
        float cur_sat=(mx>1e-5f)?(mx-mn)/mx:0.f;
        // Vibrance: less boost for already-saturated pixels
        float vib_boost=vibrance*(1.f-cur_sat);
        float total_sat=saturation+vib_boost;
        // Apply saturation
        p[0]=clampf(lm+(p[0]-lm)*(1.f+total_sat));
        p[1]=clampf(lm+(p[1]-lm)*(1.f+total_sat));
        p[2]=clampf(lm+(p[2]-lm)*(1.f+total_sat));
    }
    return to8u(f32);
}

// --- 2.6 Photo Filter (color tint over frame) ---
static cv::Mat fx_photo_filter(const cv::Mat& f,
    cv::Vec3f color_bgr,float density,bool preserve_luma){
    cv::Mat f32=to32f(f);
    cv::Mat result(f32.size(),CV_32FC3);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        float orig_luma=luma_bgr(p);
        r[0]=clampf(p[0]*(1-density)+color_bgr[0]*density);
        r[1]=clampf(p[1]*(1-density)+color_bgr[1]*density);
        r[2]=clampf(p[2]*(1-density)+color_bgr[2]*density);
        if(preserve_luma){
            float new_luma=luma_bgr(r);
            if(new_luma>1e-5f){
                float scale=orig_luma/new_luma;
                r[0]=clampf(r[0]*scale);
                r[1]=clampf(r[1]*scale);
                r[2]=clampf(r[2]*scale);
            }
        }
    }
    return to8u(result);
}

// --- 2.7 Shadows/Highlights ---
static cv::Mat fx_shadows_highlights(const cv::Mat& f,
    float shadow_lift,float highlight_roll,float contrast){
    cv::Mat f32=to32f(f);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        for(int c=0;c<3;c++){
            float v=p[c];
            // Lift shadows
            if(shadow_lift>0&&v<0.5f){
                float w=powf(1.f-v*2,1.5f);
                v=clampf(v+shadow_lift*w*0.3f);
            }
            // Roll highlights
            if(highlight_roll>0&&v>0.5f){
                float w=powf((v-0.5f)*2,1.5f);
                v=clampf(v-highlight_roll*w*0.2f);
            }
            // Contrast
            v=clampf(0.5f+(v-0.5f)*(1.f+contrast));
            p[c]=v;
        }
    }
    return to8u(f32);
}

// --- 2.8 Tint (convert to two-color tint) ---
static cv::Mat fx_tint(const cv::Mat& f,
    cv::Vec3f shadow_color,cv::Vec3f hi_color,float amount){
    cv::Mat f32=to32f(f);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        float lm=luma_bgr(p);
        cv::Vec3f tinted={
            lerp(shadow_color[0],hi_color[0],lm),
            lerp(shadow_color[1],hi_color[1],lm),
            lerp(shadow_color[2],hi_color[2],lm)
        };
        p[0]=clampf(lerp(p[0],tinted[0],amount));
        p[1]=clampf(lerp(p[1],tinted[1],amount));
        p[2]=clampf(lerp(p[2],tinted[2],amount));
    }
    return to8u(f32);
}

// --- 2.9 Three-Way Color Corrector (Lumetri-style) ---
static cv::Mat fx_three_way_color(const cv::Mat& f,
    cv::Vec3f lift,   // adds to shadows
    cv::Vec3f gamma,  // multiplies midtones
    cv::Vec3f gain){  // multiplies highlights
    cv::Mat f32=to32f(f);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        float lm=luma_bgr(p);
        float sw=powf(1-lm*2.f>0?1-lm*2.f:0,1.5f);  // shadow
        float hw=powf(lm*2.f-1>0?lm*2.f-1:0,1.5f);   // highlight
        float mw=1.f-sw-hw;
        for(int c=0;c<3;c++){
            float v=p[c];
            v += lift[c]*sw;
            v *= 1.f+gamma[c]*mw;
            v *= 1.f+gain[c]*hw;
            p[c]=clampf(v);
        }
    }
    return to8u(f32);
}

// --- 2.10 Advanced LUT System (33³) ---
struct LUT3D {
    static const int SZ=33;
    float d[SZ][SZ][SZ][3];
    void identity(){
        float step=1.f/(SZ-1);
        for(int r=0;r<SZ;r++)for(int g=0;g<SZ;g++)for(int b=0;b<SZ;b++){
            d[r][g][b][0]=r*step;
            d[r][g][b][1]=g*step;
            d[r][g][b][2]=b*step;
        }
    }
    void apply(cv::Mat& f) const {
        float sc=(SZ-1)/255.f;
        for(int y=0;y<f.rows;y++){
            auto* row=f.ptr<uchar>(y);
            for(int x=0;x<f.cols;x++){
                float bv=row[x*3+0]*sc, gv=row[x*3+1]*sc, rv=row[x*3+2]*sc;
                int r0=clamp((int)rv,0,SZ-2),g0=clamp((int)gv,0,SZ-2),b0=clamp((int)bv,0,SZ-2);
                float fr=rv-r0,fg=gv-g0,fb=bv-b0;
                float ro=0,go=0,bo=0;
                for(int dr=0;dr<2;dr++)for(int dg=0;dg<2;dg++)for(int db=0;db<2;db++){
                    float w=(dr?fr:1-fr)*(dg?fg:1-fg)*(db?fb:1-fb);
                    ro+=w*d[r0+dr][g0+dg][b0+db][0];
                    go+=w*d[r0+dr][g0+dg][b0+db][1];
                    bo+=w*d[r0+dr][g0+dg][b0+db][2];
                }
                row[x*3+2]=clamp((int)(ro*255+.5f),0,255);
                row[x*3+1]=clamp((int)(go*255+.5f),0,255);
                row[x*3+0]=clamp((int)(bo*255+.5f),0,255);
            }
        }
    }
};

// Build LUTs for every mood
static LUT3D build_lut(const std::string& mood,float intensity){
    LUT3D lut; lut.identity();
    float s=1.f/(LUT3D::SZ-1);
    for(int ri=0;ri<LUT3D::SZ;ri++)
    for(int gi=0;gi<LUT3D::SZ;gi++)
    for(int bi=0;bi<LUT3D::SZ;bi++){
        float rv=ri*s,gv=gi*s,bv=bi*s;
        float lm=0.299f*rv+0.587f*gv+0.114f*bv;

        auto& out=lut.d[ri][gi][bi];

        if(mood=="hype"||mood=="sunset_gold"){
            // Orange-gold, punchy, crushed blacks
            float hw=powf(clampf((lm-0.55f)/0.45f),1.5f);
            float sw=powf(clampf((0.40f-lm)/0.40f),1.5f);
            out[2]=clampf(rv+hw*0.14f*intensity+sw*0.04f*intensity); // R
            out[1]=clampf(gv+hw*0.05f*intensity-sw*0.03f*intensity); // G
            out[0]=clampf(bv-hw*0.12f*intensity-sw*0.06f*intensity); // B
            // Contrast
            float mid=0.5f,cont=1+0.30f*intensity;
            for(int c=0;c<3;c++) out[c]=clampf(mid+(out[c]-mid)*cont);
        }
        else if(mood=="dark"||mood=="moody"){
            float sw=powf(clampf((0.45f-lm)/0.45f),1.8f);
            float ro=clampf(rv-0.08f*intensity*(1-rv));
            float go=clampf(gv-0.08f*intensity*(1-gv));
            float bo=clampf(bv-0.08f*intensity*(1-bv));
            bo=clampf(bo+sw*0.12f*intensity);
            ro=clampf(ro-sw*0.06f*intensity);
            float d=0.30f*intensity;
            float lm2=0.299f*ro+0.587f*go+0.114f*bo;
            out[2]=clampf(ro*(1-d)+lm2*d);
            out[1]=clampf(go*(1-d)+lm2*d);
            out[0]=clampf(bo*(1-d)+lm2*d);
        }
        else if(mood=="dramatic"||mood=="teal_orange"){
            float sw=powf(clampf((0.35f-lm)/0.35f),1.6f);
            float hw=powf(clampf((lm-0.62f)/0.38f),1.4f);
            out[2]=clampf(rv+(hw*0.12f-sw*0.04f)*intensity);
            out[1]=clampf(gv+(sw*0.05f-hw*0.03f)*intensity);
            out[0]=clampf(bv+(sw*0.14f-hw*0.10f)*intensity);
        }
        else if(mood=="sad"||mood=="cold_blue"){
            float d=0.40f*intensity;
            out[2]=clampf(rv*(1-d)+lm*d-0.05f*intensity);
            out[1]=clampf(gv*(1-d)+lm*d+0.01f*intensity);
            out[0]=clampf(bv*(1-d)+lm*d+0.10f*intensity);
        }
        else if(mood=="cyberpunk"){
            float sw=powf(clampf((0.4f-lm)/0.4f),1.4f);
            float hw=powf(clampf((lm-0.58f)/0.42f),1.2f);
            float ro=clampf(rv+(sw*0.10f-hw*0.09f)*intensity);
            float go=clampf(gv-(sw*0.07f+hw*0.04f)*intensity);
            float bo=clampf(bv+(sw*0.08f+hw*0.12f)*intensity);
            float sat=1+0.35f*intensity;
            float lm2=0.299f*ro+0.587f*go+0.114f*bo;
            out[2]=clampf(lm2+(ro-lm2)*sat);
            out[1]=clampf(lm2+(go-lm2)*sat);
            out[0]=clampf(lm2+(bo-lm2)*sat);
        }
        else if(mood=="neon"){
            float sat=1+intensity*0.85f;
            float ro=clampf(lm+(rv-lm)*sat);
            float go=clampf(lm+(gv-lm)*sat);
            float bo=clampf(lm+(bv-lm)*sat);
            auto push=[&](float ch){
                float dd=ch-lm;
                return clampf(lm+dd*(1+0.4f*intensity*std::abs(dd)*3));
            };
            out[2]=push(ro); out[1]=push(go); out[0]=push(bo);
        }
        else if(mood=="bleach_bypass"){
            float d=0.55f*intensity;
            float ro=rv*(1-d)+lm*d;
            float go=gv*(1-d)+lm*d;
            float bo=bv*(1-d)+lm*d;
            auto sc=[](float x){return x*x*(3-2*x);};
            float a=intensity*0.6f;
            out[2]=clampf(ro*(1-a)+sc(ro)*a+0.025f*intensity);
            out[1]=clampf(go*(1-a)+sc(go)*a+0.025f*intensity);
            out[0]=clampf(bo*(1-a)+sc(bo)*a+0.025f*intensity);
        }
        else if(mood=="cross_process"){
            auto cr=[](float x){return clampf(x+0.08f*(float)std::sin(x*(float)M_PI));};
            auto cg=[](float x){return clampf(x+0.12f*(1-x)*x*2);};
            auto cb=[](float x){return clampf(x-0.15f*x*x);};
            out[2]=lerp(rv,cr(rv),intensity);
            out[1]=lerp(gv,cg(gv),intensity);
            out[0]=lerp(bv,cb(bv),intensity);
        }
        else if(mood=="warm"||mood=="filmic_warm"){
            // Golden/warm film look
            out[2]=clampf(rv+0.08f*intensity);
            out[1]=clampf(gv+0.03f*intensity);
            out[0]=clampf(bv-0.07f*intensity);
            // S-curve
            auto sc=[](float x,float a){return x/(1+(1-x)*a);};
            for(int c=0;c<3;c++) out[c]=clampf(sc(out[c],0.3f*intensity));
        }
        else if(mood=="chill"||mood=="anime_vibrant"){
            // Lifted blacks + saturated midtones
            float lift=0.04f*intensity;
            float rv2=rv*(1-lift)+lift;
            float gv2=gv*(1-lift)+lift;
            float bv2=bv*(1-lift)+lift;
            float mid=1+0.45f*(1-std::abs(lm*2-1))*intensity;
            out[2]=clampf(lm+(rv2-lm)*mid);
            out[1]=clampf(lm+(gv2-lm)*mid);
            out[0]=clampf(lm+(bv2-lm)*mid);
        }
        else { out[0]=bv; out[1]=gv; out[2]=rv; } // identity fallback
    }
    return lut;
}

// =============================================================================
// SECTION 3 - DISTORT EFFECTS
// =============================================================================

// --- 3.1 Turbulent Displace (AE's most-used distort) ---
static cv::Mat fx_turbulent_displace(const cv::Mat& f,
    float amount,float size,float evolution,int seed){
    int h=f.rows,w=f.cols;
    // Perlin-like noise using multi-octave OpenCV fractal
    cv::Mat noise_x(h,w,CV_32F), noise_y(h,w,CV_32F);
    std::mt19937 rng(seed+(int)(evolution*1000));
    std::uniform_real_distribution<float> ud(0,1000);
    float ox=ud(rng),oy=ud(rng);
    float freq=1.f/std::max(1.f,size);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        // Multi-octave noise approximation
        float nx=0,ny=0,amp=1,f2=freq;
        for(int oct=0;oct<4;oct++){
            // Use sin-based approximation for Perlin
            float px=(x+ox)*f2, py=(y+oy)*f2;
            float ev=evolution*(1+oct*0.5f);
            nx+=amp*(std::sin(px*1.3f+ev)*std::cos(py*0.7f-ev*0.6f));
            ny+=amp*(std::cos(px*0.9f-ev*0.8f)*std::sin(py*1.1f+ev*0.5f));
            amp*=0.5f; f2*=2.f;
        }
        noise_x.at<float>(y,x)=nx*amount;
        noise_y.at<float>(y,x)=ny*amount;
    }
    // Build remap
    cv::Mat map_x(h,w,CV_32F),map_y(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        map_x.at<float>(y,x)=clamp((float)x+noise_x.at<float>(y,x),0.f,(float)(w-1));
        map_y.at<float>(y,x)=clamp((float)y+noise_y.at<float>(y,x),0.f,(float)(h-1));
    }
    cv::Mat r;
    cv::remap(f,r,map_x,map_y,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// --- 3.2 Wave Warp ---
static cv::Mat fx_wave_warp(const cv::Mat& f,
    float amplitude,float wavelength,float phase,float direction_deg){
    int h=f.rows,w=f.cols;
    float rad=direction_deg*(float)M_PI/180;
    float dx=std::cos(rad),dy=std::sin(rad);
    float freq=2*(float)M_PI/std::max(1.f,wavelength);
    cv::Mat map_x(h,w,CV_32F),map_y(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float proj=x*dx+y*dy;
        float disp=amplitude*std::sin(freq*proj+phase);
        map_x.at<float>(y,x)=clamp((float)x+disp*dy,0.f,(float)(w-1));
        map_y.at<float>(y,x)=clamp((float)y-disp*dx,0.f,(float)(h-1));
    }
    cv::Mat r; cv::remap(f,r,map_x,map_y,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// --- 3.3 Ripple ---
static cv::Mat fx_ripple(const cv::Mat& f,
    float radius,float wave_width,float phase,float cx_f,float cy_f){
    int h=f.rows,w=f.cols;
    float cx=cx_f*w,cy=cy_f*h;
    cv::Mat map_x(h,w,CV_32F),map_y(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float dx=(float)x-cx, dy=(float)y-cy;
        float dist=std::sqrt(dx*dx+dy*dy);
        float disp=0;
        if(dist>0&&dist<radius){
            float t=dist/std::max(1.f,wave_width);
            disp=radius*0.05f*std::sin(2*(float)M_PI*t+phase);
            float nx=dx/dist, ny=dy/dist;
            map_x.at<float>(y,x)=clamp((float)x-nx*disp,0.f,(float)(w-1));
            map_y.at<float>(y,x)=clamp((float)y-ny*disp,0.f,(float)(h-1));
        } else {
            map_x.at<float>(y,x)=(float)x;
            map_y.at<float>(y,x)=(float)y;
        }
    }
    cv::Mat r; cv::remap(f,r,map_x,map_y,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// --- 3.4 Bulge ---
static cv::Mat fx_bulge(const cv::Mat& f,
    float bulge_amount,float radius_f,float cx_f,float cy_f){
    int h=f.rows,w=f.cols;
    float cx=cx_f*w,cy=cy_f*h;
    float R=radius_f*std::min(w,h)*0.5f;
    cv::Mat map_x(h,w,CV_32F),map_y(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float dx=(float)x-cx, dy=(float)y-cy;
        float dist=std::sqrt(dx*dx+dy*dy);
        if(dist<R&&dist>0){
            float t=dist/R;
            float bulge=1.f+(bulge_amount)*(1.f-t*t);
            map_x.at<float>(y,x)=clamp(cx+dx/bulge,0.f,(float)(w-1));
            map_y.at<float>(y,x)=clamp(cy+dy/bulge,0.f,(float)(h-1));
        } else {
            map_x.at<float>(y,x)=(float)x;
            map_y.at<float>(y,x)=(float)y;
        }
    }
    cv::Mat r; cv::remap(f,r,map_x,map_y,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// --- 3.5 Twirl ---
static cv::Mat fx_twirl(const cv::Mat& f,
    float angle_deg,float radius_f,float cx_f,float cy_f){
    int h=f.rows,w=f.cols;
    float cx=cx_f*w,cy=cy_f*h;
    float R=radius_f*std::min(w,h)*0.5f;
    float angle_rad=angle_deg*(float)M_PI/180;
    cv::Mat map_x(h,w,CV_32F),map_y(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float dx=(float)x-cx, dy=(float)y-cy;
        float dist=std::sqrt(dx*dx+dy*dy);
        float rot=0;
        if(dist<R) rot=angle_rad*(1.f-dist/R);
        float cs=std::cos(rot),sn=std::sin(rot);
        map_x.at<float>(y,x)=clamp(cx+dx*cs-dy*sn,0.f,(float)(w-1));
        map_y.at<float>(y,x)=clamp(cy+dx*sn+dy*cs,0.f,(float)(h-1));
    }
    cv::Mat r; cv::remap(f,r,map_x,map_y,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// --- 3.6 Spherize ---
static cv::Mat fx_spherize(const cv::Mat& f,float amount,float cx_f,float cy_f){
    int h=f.rows,w=f.cols;
    float cx=cx_f*w,cy=cy_f*h;
    float rx=w/2.f,ry=h/2.f;
    cv::Mat map_x(h,w,CV_32F),map_y(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float nx=((float)x-cx)/rx, ny=((float)y-cy)/ry;
        float r2=nx*nx+ny*ny;
        if(r2<1.f){
            float r=std::sqrt(r2);
            float theta=std::asin(r);
            float scale=(r>0)?theta/(r*(float)M_PI*0.5f):1.f;
            scale=lerp(1.f,scale,amount);
            map_x.at<float>(y,x)=clamp(cx+nx*scale*rx,0.f,(float)(w-1));
            map_y.at<float>(y,x)=clamp(cy+ny*scale*ry,0.f,(float)(h-1));
        } else {
            map_x.at<float>(y,x)=(float)x;
            map_y.at<float>(y,x)=(float)y;
        }
    }
    cv::Mat r; cv::remap(f,r,map_x,map_y,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// --- 3.7 Mirror ---
static cv::Mat fx_mirror(const cv::Mat& f,float angle_deg,float cx_f,float cy_f){
    int h=f.rows,w=f.cols;
    float cx=cx_f*w,cy=cy_f*h;
    float rad=angle_deg*(float)M_PI/180;
    float nx=std::sin(rad),ny=-std::cos(rad); // normal to mirror line
    cv::Mat map_x(h,w,CV_32F),map_y(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float dx=(float)x-cx, dy=(float)y-cy;
        float dot=dx*nx+dy*ny;
        if(dot>=0){
            // Reflect
            map_x.at<float>(y,x)=clamp((float)x-2*dot*nx,0.f,(float)(w-1));
            map_y.at<float>(y,x)=clamp((float)y-2*dot*ny,0.f,(float)(h-1));
        } else {
            map_x.at<float>(y,x)=(float)x;
            map_y.at<float>(y,x)=(float)y;
        }
    }
    cv::Mat r; cv::remap(f,r,map_x,map_y,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// --- 3.8 Polar Coordinates ---
static cv::Mat fx_polar_coordinates(const cv::Mat& f,float amount,bool rect_to_polar){
    int h=f.rows,w=f.cols;
    cv::Mat map_x(h,w,CV_32F),map_y(h,w,CV_32F);
    float cx=w/2.f,cy=h/2.f;
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        if(rect_to_polar){
            float dx=(float)x-cx,dy=(float)y-cy;
            float r=std::sqrt(dx*dx+dy*dy);
            float theta=std::atan2(dy,dx)+(float)M_PI;
            float u=(theta/(2*(float)M_PI))*w;
            float v=(r/std::sqrt(cx*cx+cy*cy))*h;
            map_x.at<float>(y,x)=lerp((float)x,clamp(u,0.f,(float)(w-1)),amount);
            map_y.at<float>(y,x)=lerp((float)y,clamp(v,0.f,(float)(h-1)),amount);
        } else {
            float theta=((float)x/w)*2*(float)M_PI-(float)M_PI;
            float r=((float)y/h)*std::sqrt(cx*cx+cy*cy);
            float u=clamp(cx+r*std::cos(theta),0.f,(float)(w-1));
            float v=clamp(cy+r*std::sin(theta),0.f,(float)(h-1));
            map_x.at<float>(y,x)=lerp((float)x,u,amount);
            map_y.at<float>(y,x)=lerp((float)y,v,amount);
        }
    }
    cv::Mat r; cv::remap(f,r,map_x,map_y,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// --- 3.9 Displacement Map ---
static cv::Mat fx_displacement_map(const cv::Mat& f,const cv::Mat& map_src,
    float max_h,float max_v,int channel=0){
    int h=f.rows,w=f.cols;
    cv::Mat map_gray;
    if(map_src.channels()==3) cv::cvtColor(map_src,map_gray,cv::COLOR_BGR2GRAY);
    else map_gray=map_src;
    cv::Mat map_r; cv::resize(map_gray,map_r,cv::Size(w,h));
    cv::Mat mx(h,w,CV_32F),my(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float v=(map_r.at<uchar>(y,x)/255.f-0.5f)*2;
        mx.at<float>(y,x)=clamp((float)x+v*max_h,0.f,(float)(w-1));
        my.at<float>(y,x)=clamp((float)y+v*max_v,0.f,(float)(h-1));
    }
    cv::Mat r; cv::remap(f,r,mx,my,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// --- 3.10 Liquify (push pixels) ---
static cv::Mat fx_liquify(const cv::Mat& f,
    float cx_f,float cy_f,float radius_f,float strength,float t){
    int h=f.rows,w=f.cols;
    float cx=cx_f*w,cy=cy_f*h,R=radius_f*std::min(w,h);
    float push_x=strength*std::cos(t)*20,push_y=strength*std::sin(t)*20;
    cv::Mat mx(h,w,CV_32F),my(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float dx=(float)x-cx,dy=(float)y-cy;
        float dist=std::sqrt(dx*dx+dy*dy);
        float falloff=(dist<R)?powf(1-dist/R,2.f):0;
        mx.at<float>(y,x)=clamp((float)x-push_x*falloff,0.f,(float)(w-1));
        my.at<float>(y,x)=clamp((float)y-push_y*falloff,0.f,(float)(h-1));
    }
    cv::Mat r; cv::remap(f,r,mx,my,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    return r;
}

// =============================================================================
// SECTION 4 - GENERATE EFFECTS
// =============================================================================

// --- 4.1 Advanced Lightning ---
static cv::Mat fx_advanced_lightning(const cv::Mat& f,
    cv::Point2f start,cv::Point2f end,float width,
    cv::Vec3b color,float forking,float t){
    cv::Mat result=f.clone();
    // Recursive bolt generator
    std::mt19937 rng((int)(t*10000));
    std::uniform_real_distribution<float> ud(-1,1);

    std::function<void(cv::Point2f,cv::Point2f,int)> draw_bolt;
    draw_bolt=[&](cv::Point2f a,cv::Point2f b,int depth){
        if(depth<=0) return;
        cv::Point2f mid={(a.x+b.x)/2+(b.y-a.y)*0.3f*ud(rng),
                         (a.y+b.y)/2+(b.x-a.x)*0.3f*ud(rng)};
        int lw=std::max(1,(int)(width*depth/5));
        cv::line(result,cv::Point((int)a.x,(int)a.y),
                 cv::Point((int)mid.x,(int)mid.y),
                 cv::Scalar(color[0],color[1],color[2]),lw,cv::LINE_AA);
        cv::line(result,cv::Point((int)mid.x,(int)mid.y),
                 cv::Point((int)b.x,(int)b.y),
                 cv::Scalar(color[0],color[1],color[2]),lw,cv::LINE_AA);
        draw_bolt(a,mid,depth-1);
        draw_bolt(mid,b,depth-1);
        // Fork
        if(depth>=3&&ud(rng)>1-forking){
            cv::Point2f fork_end={
                mid.x+(b.x-a.x)*0.5f+ud(rng)*50,
                mid.y+(b.y-a.y)*0.5f+ud(rng)*50};
            draw_bolt(mid,fork_end,depth-2);
        }
    };
    draw_bolt(start,end,5);
    // Glow
    cv::Mat glow; cv::GaussianBlur(result,glow,cv::Size(0,0),width*2);
    cv::addWeighted(result,1.0,glow,0.5,0,result);
    return result;
}

// --- 4.2 Lens Flare ---
static cv::Mat fx_lens_flare(const cv::Mat& f,
    float cx_f,float cy_f,float brightness,float size){
    cv::Mat result=f.clone();
    int h=f.rows,w=f.cols;
    float cx=cx_f*w,cy=cy_f*h;
    float rs=size*std::min(w,h)*0.1f;

    // Main flare circles
    struct FlareElem { float dx,dy,r,alpha; cv::Vec3b col; };
    std::vector<FlareElem> elems={
        {0,0,rs*2.5f,0.9f,{200,220,255}},
        {0,0,rs*0.8f,0.8f,{255,240,200}},
        {0.3f,0.3f,rs*0.4f,0.5f,{180,200,255}},
        {-0.6f,-0.6f,rs*0.6f,0.4f,{255,200,180}},
        {1.2f,1.2f,rs*0.3f,0.6f,{200,255,200}},
        {-1.5f,-1.5f,rs*0.5f,0.35f,{255,180,255}},
        // Anamorphic streaks
    };
    // Center the lens on the frame
    float dx_center=w/2.f-cx, dy_center=h/2.f-cy;

    for(auto& e:elems){
        cv::Point ctr((int)(cx+e.dx*dx_center),(int)(cy+e.dy*dy_center));
        cv::Mat flare_layer=cv::Mat::zeros(h,w,CV_8UC3);
        cv::circle(flare_layer,ctr,(int)e.r,
                   cv::Scalar(e.col[0],e.col[1],e.col[2]),-1);
        int blur_r=(int)(e.r*0.8f); if(blur_r%2==0)blur_r++;if(blur_r<3)blur_r=3;
        cv::GaussianBlur(flare_layer,flare_layer,cv::Size(blur_r,blur_r),e.r/3);
        cv::addWeighted(result,1.0,flare_layer,e.alpha*brightness,0,result);
    }
    // Anamorphic horizontal streak
    cv::Mat streak=cv::Mat::zeros(h,w,CV_8UC3);
    for(int x=0;x<w;x++){
        float d=std::abs(x-cx);
        float a=brightness*0.5f*std::exp(-d*d/(2*w*w*0.02f));
        int av=(int)(a*255);
        streak.at<cv::Vec3b>((int)cy,x)={
            (uchar)std::min(255,av+80),
            (uchar)std::min(255,av+100),
            (uchar)std::min(255,av+120)};
    }
    int sk=(int)(size*15); if(sk%2==0)sk++; if(sk<3)sk=3;
    cv::GaussianBlur(streak,streak,cv::Size(w|1,sk),w*0.03f,size*3);
    cv::addWeighted(result,1.0,streak,0.7f*brightness,0,result);
    return result;
}

// --- 4.3 Fractal Noise ---
static cv::Mat fx_fractal_noise(const cv::Mat& f,
    float scale,float complexity,float evolution,float opacity,int seed){
    int h=f.rows,w=f.cols;
    cv::Mat noise=cv::Mat::zeros(h,w,CV_32F);
    float amp=1.f, freq=1.f/std::max(1.f,scale);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> od(0,1000);
    for(int oct=0;oct<(int)complexity;oct++){
        float ox=od(rng),oy=od(rng),oz=evolution*0.5f*oct;
        for(int y=0;y<h;y++)
        for(int x=0;x<w;x++){
            float px=(x+ox)*freq,py=(y+oy)*freq;
            float n=std::sin(px*1.2f+oz)*std::cos(py*0.8f+oz*0.7f)
                   *0.5f+0.5f;
            n+=0.5f*std::sin(px*2.3f-oz*1.3f)*std::cos(py*1.7f+oz);
            noise.at<float>(y,x)+=amp*n;
        }
        amp*=0.5f; freq*=2.f;
    }
    cv::normalize(noise,noise,0,1,cv::NORM_MINMAX);
    // Overlay noise on frame
    cv::Mat f32=to32f(f);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float n=noise.at<float>(y,x);
        auto& p=f32.at<cv::Vec3f>(y,x);
        p[0]=clampf(lerp(p[0],n,opacity));
        p[1]=clampf(lerp(p[1],n,opacity));
        p[2]=clampf(lerp(p[2],n,opacity));
    }
    return to8u(f32);
}

// --- 4.4 Cell Pattern ---
static cv::Mat fx_cell_pattern(const cv::Mat& f,
    float size,float softness,float evolution,float opacity,int seed){
    int h=f.rows,w=f.cols;
    int cells_x=std::max(2,(int)(w/size));
    int cells_y=std::max(2,(int)(h/size));
    // Random cell centers
    std::mt19937 rng(seed+(int)(evolution*1000));
    std::uniform_real_distribution<float> ud(0,1);
    std::vector<cv::Point2f> centers;
    for(int i=0;i<cells_x*cells_y;i++){
        centers.push_back({ud(rng)*(float)w,ud(rng)*(float)h});
    }
    cv::Mat cell_map(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float min_d=1e9f,min2=1e9f;
        for(auto& c:centers){
            float d=std::sqrt(powf(x-c.x,2)+powf(y-c.y,2));
            if(d<min_d){min2=min_d;min_d=d;}
            else if(d<min2) min2=d;
        }
        // Voronoi edge distance
        float edge_dist=min2-min_d;
        cell_map.at<float>(y,x)=clampf(edge_dist/softness);
    }
    cv::Mat f32=to32f(f);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float n=cell_map.at<float>(y,x);
        auto& p=f32.at<cv::Vec3f>(y,x);
        p[0]=clampf(lerp(p[0],n,opacity));
        p[1]=clampf(lerp(p[1],n,opacity));
        p[2]=clampf(lerp(p[2],n,opacity));
    }
    return to8u(f32);
}

// --- 4.5 Particle Systems ---
struct Particle {
    float x,y,vx,vy,life,max_life,size;
    cv::Vec3b color;
};

static cv::Mat fx_particles(const cv::Mat& f,
    float t,std::vector<Particle>& particles,
    float gravity,float wind,const std::string& type){
    cv::Mat result=f.clone();
    // Spawn new particles
    std::mt19937 rng((int)(t*1000)^42);
    std::uniform_real_distribution<float> ud(-1,1);
    int h=f.rows,w=f.cols;
    int n_spawn=20;
    for(int i=0;i<n_spawn&&(int)particles.size()<800;i++){
        Particle p;
        if(type=="rain"){
            p.x=ud(rng)*0.5f*w+w/2;p.y=-10;
            p.vx=wind+ud(rng)*2;p.vy=8+ud(rng)*4;
            p.life=p.max_life=h/(p.vy);
            p.size=1.5f;p.color={200,210,255};
        } else if(type=="snow"){
            p.x=ud(rng)*0.6f*w+w/2;p.y=-10;
            p.vx=(ud(rng)*1.5f)+wind;p.vy=1+ud(rng)*1.5f;
            p.life=p.max_life=h/p.vy;
            p.size=2+ud(rng)*2;p.color={230,235,255};
        } else if(type=="sparks"){
            p.x=w/2+ud(rng)*50;p.y=h/2;
            p.vx=ud(rng)*8;p.vy=ud(rng)*8;
            p.life=p.max_life=1+ud(rng)*0.8f;
            p.size=2+ud(rng)*2;
            p.color={(uchar)255,(uchar)(150+ud(rng)*100),(uchar)(50+ud(rng)*80)};
        } else {  // smoke/dust
            p.x=ud(rng)*w;p.y=h;
            p.vx=ud(rng)*2+wind;p.vy=-1.5f-ud(rng)*1;
            p.life=p.max_life=60;
            p.size=4+ud(rng)*8;p.color={180,180,185};
        }
        particles.push_back(p);
    }
    // Update and draw
    for(auto it=particles.begin();it!=particles.end();){
        it->x+=it->vx; it->y+=it->vy;
        it->vy+=gravity; it->vx+=wind*0.1f;
        it->life-=1;
        float alpha=it->life/it->max_life;
        if(it->life<=0||it->x<0||it->x>w||it->y>h){
            it=particles.erase(it); continue;
        }
        int x=(int)it->x,y=(int)it->y;
        int r=(int)it->size;
        if(type=="rain"){
            cv::line(result,{x,y},{x+(int)(it->vx*2),y+(int)(it->vy*0.5f)},
                cv::Scalar(it->color[0],it->color[1],it->color[2]),1,cv::LINE_AA);
        } else {
            cv::circle(result,{x,y},r,
                cv::Scalar(it->color[0]*alpha,it->color[1]*alpha,it->color[2]*alpha),
                -1,cv::LINE_AA);
            if(type=="sparks"||type=="snow"){
                // Glow
                int gr=r*2; if(gr<3)gr=3; if(gr%2==0)gr++;
                cv::Mat gl=cv::Mat::zeros(result.rows,result.cols,CV_8UC3);
                cv::circle(gl,{x,y},gr,cv::Scalar(it->color[0],it->color[1],it->color[2]),-1);
                cv::GaussianBlur(gl,gl,cv::Size(gr,gr),r);
                cv::addWeighted(result,1,gl,alpha*0.6f,0,result);
            }
        }
        ++it;
    }
    return result;
}

// =============================================================================
// SECTION 5 - STYLIZE EFFECTS
// =============================================================================

// --- 5.1 Glow (multi-pass, proper) ---
static cv::Mat fx_glow(const cv::Mat& f,float threshold,float radius,float intensity){
    cv::Mat f32=to32f(f);
    cv::Mat bright(f32.size(),CV_32FC3);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        float lm=luma_bgr(p);
        float mask=clampf((lm-threshold)/std::max(0.01f,1-threshold));
        mask=mask*mask; // sharper threshold
        bright.at<cv::Vec3f>(y,x)=p*mask;
    }
    // Multi-scale bloom
    cv::Mat bloom=cv::Mat::zeros(f32.size(),CV_32FC3);
    float scales[]={1.f,2.5f,6.f};
    float weights[]={0.5f,0.3f,0.2f};
    int br=std::max(3,(int)(std::min(f32.cols,f32.rows)*0.012f*radius));
    for(int i=0;i<3;i++){
        int r=(int)(br*scales[i]); if(r%2==0)r++; if(r<3)r=3;
        cv::Mat bl; cv::GaussianBlur(bright,bl,cv::Size(r,r),r/3.0);
        cv::addWeighted(bloom,1,bl,weights[i],0,bloom);
    }
    bloom*=intensity;
    // Screen blend
    cv::Mat result(f32.size(),CV_32FC3);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& b=f32.at<cv::Vec3f>(y,x);
        auto& g=bloom.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        r[0]=clampf(1-(1-b[0])*(1-clampf(g[0])));
        r[1]=clampf(1-(1-b[1])*(1-clampf(g[1])));
        r[2]=clampf(1-(1-b[2])*(1-clampf(g[2])));
    }
    return to8u(result);
}

// --- 5.2 Find Edges ---
static cv::Mat fx_find_edges(const cv::Mat& f,float strength,bool invert){
    cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Laplacian(gray,edges,CV_16S,3);
    cv::convertScaleAbs(edges,edges);
    cv::Mat edges3; cv::cvtColor(edges,edges3,cv::COLOR_GRAY2BGR);
    if(invert) cv::bitwise_not(edges3,edges3);
    cv::Mat result;
    cv::addWeighted(f,1-strength,edges3,strength,0,result);
    return result;
}

// --- 5.3 Cartoon (AE's Cartoon effect) ---
static cv::Mat fx_cartoon(const cv::Mat& f,int detail_radius,
    float detail_thresh,int colors,float edge_strength){
    // Step 1: Smooth (remove small detail)
    cv::Mat smooth;
    cv::bilateralFilter(f,smooth,detail_radius*2,75,75);
    // Step 2: Posterize
    cv::Mat post=smooth.clone();
    for(int y=0;y<post.rows;y++)
    for(int x=0;x<post.cols;x++){
        auto& p=post.at<cv::Vec3b>(y,x);
        for(int c=0;c<3;c++){
            p[c]=(uchar)(std::round(p[c]/(255.f/colors))*(255.f/colors));
        }
    }
    // Step 3: Find edges
    cv::Mat gray; cv::cvtColor(smooth,gray,cv::COLOR_BGR2GRAY);
    cv::Mat edges; cv::Canny(gray,edges,detail_thresh*0.5f,detail_thresh);
    cv::dilate(edges,edges,cv::Mat(),cv::Point(-1,-1),1);
    // Step 4: Apply edges as black outlines
    cv::Mat result=post.clone();
    for(int y=0;y<result.rows;y++)
    for(int x=0;x<result.cols;x++){
        if(edges.at<uchar>(y,x)>0){
            auto& p=result.at<cv::Vec3b>(y,x);
            float darken=1-edge_strength;
            p[0]=(uchar)(p[0]*darken);p[1]=(uchar)(p[1]*darken);p[2]=(uchar)(p[2]*darken);
        }
    }
    return result;
}

// --- 5.4 Emboss ---
static cv::Mat fx_emboss(const cv::Mat& f,float direction_deg,float depth,float relief){
    cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
    float rad=direction_deg*(float)M_PI/180;
    float dx=std::cos(rad), dy=std::sin(rad);
    cv::Mat kernel=(cv::Mat_<float>(3,3)<<
        -depth*dx-depth*dy, -depth*dy,         depth*dx-depth*dy,
        -depth*dx,           0,                  depth*dx,
         depth*dx-depth*dy,  depth*dy,          depth*dx+depth*dy);
    kernel/=cv::sum(cv::abs(kernel))[0];
    cv::Mat embossed; cv::filter2D(gray,embossed,CV_32F,kernel);
    cv::normalize(embossed,embossed,0,255,cv::NORM_MINMAX);
    embossed.convertTo(embossed,CV_8U);
    cv::Mat embossed3; cv::cvtColor(embossed,embossed3,cv::COLOR_GRAY2BGR);
    cv::Mat result;
    cv::addWeighted(f,1-relief,embossed3,relief,0,result);
    return result;
}

// --- 5.5 Mosaic ---
static cv::Mat fx_mosaic(const cv::Mat& f,int block_w,int block_h){
    cv::Mat result=f.clone();
    int h=f.rows,w=f.cols;
    for(int y=0;y<h;y+=block_h)
    for(int x=0;x<w;x+=block_w){
        int y2=std::min(y+block_h,h),x2=std::min(x+block_w,w);
        cv::Rect roi(x,y,x2-x,y2-y);
        cv::Scalar mean=cv::mean(f(roi));
        result(roi).setTo(mean);
    }
    return result;
}

// --- 5.6 Posterize ---
static cv::Mat fx_posterize(const cv::Mat& f,int levels){
    cv::Mat result=f.clone();
    float step=255.f/levels;
    for(int y=0;y<result.rows;y++)
    for(int x=0;x<result.cols;x++){
        auto& p=result.at<cv::Vec3b>(y,x);
        for(int c=0;c<3;c++)
            p[c]=(uchar)(std::round(p[c]/step)*step);
    }
    return result;
}

// --- 5.7 Scatter ---
static cv::Mat fx_scatter(const cv::Mat& f,int amount,int seed){
    cv::Mat result=f.clone();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-amount,amount);
    for(int y=0;y<f.rows;y++)
    for(int x=0;x<f.cols;x++){
        int nx=clamp(x+dist(rng),0,f.cols-1);
        int ny=clamp(y+dist(rng),0,f.rows-1);
        result.at<cv::Vec3b>(y,x)=f.at<cv::Vec3b>(ny,nx);
    }
    return result;
}

// --- 5.8 Roughen Edges ---
static cv::Mat fx_roughen_edges(const cv::Mat& f,float amount,float evolution,int seed){
    cv::Mat result=f.clone();
    int h=f.rows,w=f.cols;
    std::mt19937 rng(seed);
    // Add turbulent displacement near edges
    cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
    cv::Mat edges; cv::Canny(gray,edges,50,150);
    cv::dilate(edges,edges,cv::Mat(),cv::Point(-1,-1),(int)(amount/10)+1);
    for(int y=1;y<h-1;y++)
    for(int x=1;x<w-1;x++){
        if(edges.at<uchar>(y,x)>0){
            float noise=std::sin(x*0.1f+evolution)*std::cos(y*0.1f+evolution*0.7f);
            int nx=clamp(x+(int)(noise*amount),0,w-1);
            int ny=clamp(y+(int)(noise*amount*0.5f),0,h-1);
            result.at<cv::Vec3b>(y,x)=f.at<cv::Vec3b>(ny,nx);
        }
    }
    return result;
}

// --- 5.9 Strobe Light ---
static cv::Mat fx_strobe_light(const cv::Mat& f,float t,float freq,float blend){
    float strobe_val=std::sin(2*(float)M_PI*freq*t);
    if(strobe_val>0){
        cv::Mat bright;
        f.convertTo(bright,-1,1+blend,0);
        return bright;
    }
    cv::Mat dark;
    f.convertTo(dark,-1,1-blend*0.5f,0);
    return dark;
}

// --- 5.10 Motion Tile ---
static cv::Mat fx_motion_tile(const cv::Mat& f,float offset_x,float offset_y,
    float phase_x,float phase_y){
    cv::Mat result=f.clone();
    int h=f.rows,w=f.cols;
    int ox=(int)(offset_x*w+phase_x)%w;
    int oy=(int)(offset_y*h+phase_y)%h;
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        int sx=(x+ox)%w; if(sx<0)sx+=w;
        int sy=(y+oy)%h; if(sy<0)sy+=h;
        result.at<cv::Vec3b>(y,x)=f.at<cv::Vec3b>(sy,sx);
    }
    return result;
}

// =============================================================================
// SECTION 6 - NOISE & GRAIN
// =============================================================================

// --- 6.1 Film Grain (multi-layer, realistic) ---
static cv::Mat fx_film_grain_v2(const cv::Mat& f,float strength,float t,int seed){
    cv::Mat f32=to32f(f);
    int h=f32.rows,w=f32.cols;
    std::mt19937 rng(seed^(int)(t*10000));
    std::normal_distribution<float> nd(0,strength*0.08f);
    // Layer 1: fine grain
    // Layer 2: larger cluster
    // Layer 3: luma variation
    cv::Mat grain1(h,w,CV_32F), grain2, grain3;
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) grain1.at<float>(y,x)=nd(rng);
    cv::GaussianBlur(grain1,grain2,cv::Size(3,3),0.8);
    cv::GaussianBlur(grain1,grain3,cv::Size(9,9),2.5);
    float pulse=0.88f+0.12f*(float)std::sin(2*(float)M_PI*4.0f*t);
    // Combine
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        float lm=luma_bgr(p);
        // Grain is less visible in very bright or very dark areas (real film)
        float sensitivity=1-std::abs(lm*2-1)*0.4f;
        float g=grain2.at<float>(y,x)*0.6f+grain3.at<float>(y,x)*0.4f;
        g*=pulse*sensitivity;
        for(int c=0;c<3;c++) p[c]=clampf(p[c]+g);
    }
    return to8u(f32);
}

// --- 6.2 Dust & Scratches ---
static cv::Mat fx_dust_scratches(const cv::Mat& f,float density,float size,int seed){
    cv::Mat result=f.clone();
    int h=f.rows,w=f.cols;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(0,1);
    // Dust specks
    int n_dust=(int)(density*w*h*0.0008f);
    for(int i=0;i<n_dust;i++){
        int x=(int)(ud(rng)*w),y=(int)(ud(rng)*h);
        int r=1+(int)(ud(rng)*size);
        int b=(int)(180+ud(rng)*75);
        cv::circle(result,{x,y},r,cv::Scalar(b,b,b),-1,cv::LINE_AA);
    }
    // Vertical scratches
    int n_scratch=(int)(density*3);
    for(int i=0;i<n_scratch;i++){
        int x=(int)(ud(rng)*w);
        int y1=(int)(ud(rng)*h*0.5f);
        int y2=(int)(ud(rng)*h*0.5f+h*0.5f);
        int thick=1+(int)(ud(rng)*1.5f);
        int b=(int)(140+ud(rng)*80);
        cv::line(result,{x,y1},{x+(int)(ud(rng)*4-2),y2},
                 cv::Scalar(b,b,b),thick,cv::LINE_AA);
    }
    return result;
}

// --- 6.3 Median ---
static cv::Mat fx_median(const cv::Mat& f,int ksize){
    cv::Mat r; cv::medianBlur(f,r,ksize%2==0?ksize+1:ksize);
    return r;
}

// =============================================================================
// SECTION 7 - TIME EFFECTS
// =============================================================================

// --- 7.1 Echo Trail ---
static cv::Mat fx_echo_trail_v2(const cv::Mat& cur,
    std::deque<cv::Mat>& buf,int n_frames,float decay,
    const std::string& blend_mode){
    buf.push_back(cur.clone());
    while((int)buf.size()>n_frames+1) buf.pop_front();
    if(buf.size()<2) return cur;

    cv::Mat accum; cur.convertTo(accum,CV_32FC3);
    float total=1.f;
    for(int i=(int)buf.size()-2;i>=0;i--){
        int age=(int)buf.size()-2-i+1;
        float w=powf(decay,(float)age);
        cv::Mat aged; buf[i].convertTo(aged,CV_32FC3);
        if(blend_mode=="screen"){
            // Screen blend the echo
            cv::Mat a32=accum/255.f;
            cv::Mat b32=aged/255.f;
            for(int y=0;y<accum.rows;y++)
            for(int x=0;x<accum.cols;x++){
                auto& pa=a32.at<cv::Vec3f>(y,x);
                auto& pb=b32.at<cv::Vec3f>(y,x);
                pa[0]=clampf(1-(1-pa[0])*(1-pb[0]*w));
                pa[1]=clampf(1-(1-pa[1])*(1-pb[1]*w));
                pa[2]=clampf(1-(1-pa[2])*(1-pb[2]*w));
            }
            accum=a32*255.f;
        } else {
            cv::addWeighted(accum,1,aged,w,0,accum);
        }
        total+=w;
    }
    if(blend_mode!="screen") accum/=total;
    cv::Mat r; accum.convertTo(r,CV_8UC3);
    return r;
}

// --- 7.2 Posterize Time (reduce effective frame rate) ---
static int posterize_frame_idx=0;
static cv::Mat posterize_last;
static cv::Mat fx_posterize_time(const cv::Mat& f,float src_fps,float target_fps){
    int every=std::max(1,(int)(src_fps/target_fps));
    if(posterize_frame_idx%every==0||posterize_last.empty())
        posterize_last=f.clone();
    posterize_frame_idx++;
    return posterize_last.clone();
}

// --- 7.3 Time Displacement ---
static cv::Mat fx_time_displacement(const cv::Mat& cur,const cv::Mat& prev,
    const cv::Mat& map_layer,float max_disp){
    int h=cur.rows,w=cur.cols;
    cv::Mat map_gray;
    if(map_layer.channels()==3) cv::cvtColor(map_layer,map_gray,cv::COLOR_BGR2GRAY);
    else map_gray=map_layer;
    cv::Mat map_r; cv::resize(map_gray,map_r,cv::Size(w,h));
    cv::Mat result=cur.clone();
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float disp=(map_r.at<uchar>(y,x)/255.f)*max_disp;
        // Blend between prev and current based on displacement
        float t=clampf(disp);
        auto& pc=cur.at<cv::Vec3b>(y,x);
        auto& pp=prev.empty()?pc:prev.at<cv::Vec3b>(y,x);
        result.at<cv::Vec3b>(y,x)={
            (uchar)lerp((float)pp[0],(float)pc[0],t),
            (uchar)lerp((float)pp[1],(float)pc[1],t),
            (uchar)lerp((float)pp[2],(float)pc[2],t)
        };
    }
    return result;
}

// =============================================================================
// SECTION 8 - SIMULATION EFFECTS
// =============================================================================

// --- 8.1 Shatter ---
static cv::Mat fx_shatter(const cv::Mat& f,float progress,int n_pieces,int seed){
    cv::Mat result=f.clone();
    int h=f.rows,w=f.cols;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(0,1);
    for(int i=0;i<n_pieces;i++){
        int x=(int)(ud(rng)*w),y=(int)(ud(rng)*h);
        int pw=20+(int)(ud(rng)*80),ph=20+(int)(ud(rng)*80);
        int x2=std::min(x+pw,w),y2=std::min(y+ph,h);
        cv::Rect roi(x,y,x2-x,y2-y);
        float vx=(ud(rng)-0.5f)*progress*40;
        float vy=(ud(rng)-0.5f)*progress*40;
        float angle=ud(rng)*progress*360;
        // Translate the piece
        cv::Mat M=cv::getRotationMatrix2D({(float)(x+pw/2),(float)(y+ph/2)},angle,1);
        M.at<double>(0,2)+=vx; M.at<double>(1,2)+=vy;
        cv::Mat piece=f(roi).clone();
        cv::Mat dest(h,w,CV_8UC3,cv::Scalar(0,0,0));
        cv::warpAffine(piece,result(roi),M(cv::Rect(0,0,3,2)),roi.size(),
                       cv::INTER_LINEAR,cv::BORDER_TRANSPARENT);
        if(progress>0.5f){
            // Fade pieces
            result(roi).convertTo(result(roi),-1,1-progress*0.5f,0);
        }
    }
    return result;
}

// --- 8.2 Caustics (underwater light effect) ---
static cv::Mat fx_caustics(const cv::Mat& f,float t,float intensity){
    int h=f.rows,w=f.cols;
    cv::Mat caustic(h,w,CV_32F);
    float speed=t*2.f;
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float u=(float)x/w, v=(float)y/h;
        float c = std::sin(u*8+speed)*std::cos(v*8-speed*0.7f)
                + std::sin(u*12-speed*0.5f)*std::cos(v*6+speed*0.8f)
                + std::sin(u*5+v*7+speed*1.2f);
        caustic.at<float>(y,x)=clampf((c+3)/6);
    }
    cv::GaussianBlur(caustic,caustic,cv::Size(7,7),2);
    cv::Mat f32=to32f(f);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float c=caustic.at<float>(y,x);
        auto& p=f32.at<cv::Vec3f>(y,x);
        // Add warm light caustics
        p[2]=clampf(p[2]+c*intensity*0.3f);
        p[1]=clampf(p[1]+c*intensity*0.25f);
        p[0]=clampf(p[0]+c*intensity*0.10f);
    }
    return to8u(f32);
}

// =============================================================================
// SECTION 9 - PERSPECTIVE EFFECTS
// =============================================================================

// --- 9.1 Drop Shadow ---
static cv::Mat fx_drop_shadow(const cv::Mat& f,int offset_x,int offset_y,
    float blur,float opacity,cv::Vec3b color){
    cv::Mat result=f.clone();
    cv::Mat shadow(f.rows,f.cols,CV_8UC3,cv::Scalar(color[0],color[1],color[2]));
    int bk=std::max(1,(int)(blur))|1;
    cv::GaussianBlur(shadow,shadow,cv::Size(bk,bk),blur/3);
    // Offset shadow and blend
    cv::Mat offset_shadow=cv::Mat::zeros(f.rows,f.cols,CV_8UC3);
    cv::Rect src_roi(std::max(0,-offset_x),std::max(0,-offset_y),
                     f.cols-std::abs(offset_x),f.rows-std::abs(offset_y));
    cv::Rect dst_roi(std::max(0,offset_x),std::max(0,offset_y),
                     src_roi.width,src_roi.height);
    if(src_roi.width>0&&src_roi.height>0&&
       dst_roi.x+dst_roi.width<=f.cols&&dst_roi.y+dst_roi.height<=f.rows){
        shadow(src_roi).copyTo(offset_shadow(dst_roi));
    }
    cv::addWeighted(result,1,offset_shadow,opacity,0,result);
    return result;
}

// --- 9.2 Bevel Edges ---
static cv::Mat fx_bevel_edges(const cv::Mat& f,int thickness,float light_angle){
    cv::Mat result=f.clone();
    float rad=light_angle*(float)M_PI/180;
    float lx=std::cos(rad),ly=std::sin(rad);
    // Top/left edges: highlight, Bottom/right: shadow
    for(int i=0;i<thickness;i++){
        float t=(float)(thickness-i)/thickness;
        int bright=(int)(255*t*std::max(0.f,lx));
        int dark=(int)(255*t*std::max(0.f,-lx));
        // Top
        cv::line(result,{0,i},{f.cols,i},cv::Scalar(bright,bright,bright),1);
        // Bottom
        cv::line(result,{0,f.rows-i-1},{f.cols,f.rows-i-1},cv::Scalar(dark,dark,dark),1);
        // Left
        cv::line(result,{i,0},{i,f.rows},cv::Scalar(bright,bright,bright),1);
        // Right
        cv::line(result,{f.cols-i-1,0},{f.cols-i-1,f.rows},cv::Scalar(dark,dark,dark),1);
    }
    return result;
}

// =============================================================================
// SECTION 10 - CHANNEL EFFECTS
// =============================================================================

// --- 10.1 Invert ---
static cv::Mat fx_invert(const cv::Mat& f,bool r,bool g,bool b){
    cv::Mat result=f.clone();
    std::vector<cv::Mat> ch; cv::split(result,ch);
    if(b) cv::bitwise_not(ch[0],ch[0]);
    if(g) cv::bitwise_not(ch[1],ch[1]);
    if(r) cv::bitwise_not(ch[2],ch[2]);
    cv::merge(ch,result);
    return result;
}

// --- 10.2 Shift Channels ---
static cv::Mat fx_shift_channels(const cv::Mat& f,int shift_r,int shift_g,int shift_b){
    std::vector<cv::Mat> ch; cv::split(f,ch);
    auto shift=[](cv::Mat& m,int s)->cv::Mat{
        if(s==0) return m.clone();
        cv::Mat r=cv::Mat::zeros(m.size(),m.type());
        if(s>0&&s<m.cols) m(cv::Rect(0,0,m.cols-s,m.rows)).copyTo(r(cv::Rect(s,0,m.cols-s,m.rows)));
        else if(s<0&&-s<m.cols) m(cv::Rect(-s,0,m.cols+s,m.rows)).copyTo(r(cv::Rect(0,0,m.cols+s,m.rows)));
        return r;
    };
    std::vector<cv::Mat> out={shift(ch[0],shift_b),shift(ch[1],shift_g),shift(ch[2],shift_r)};
    cv::Mat result; cv::merge(out,result);
    return result;
}

// --- 10.3 Minimax ---
static cv::Mat fx_minimax(const cv::Mat& f,int radius,bool use_max){
    cv::Mat result;
    if(use_max) cv::dilate(f,result,cv::Mat(),cv::Point(-1,-1),radius);
    else        cv::erode(f,result,cv::Mat(),cv::Point(-1,-1),radius);
    return result;
}

// --- 10.4 Blend ---
static cv::Mat fx_blend(const cv::Mat& f,const cv::Mat& bg,float amount,const std::string& mode){
    cv::Mat a32=to32f(f), b32;
    cv::Mat bg_r; cv::resize(bg,bg_r,f.size());
    b32=to32f(bg_r);
    cv::Mat result(f.size(),CV_32FC3);
    for(int y=0;y<f.rows;y++)
    for(int x=0;x<f.cols;x++){
        auto& a=a32.at<cv::Vec3f>(y,x);
        auto& b=b32.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        cv::Vec3f mixed;
        if(mode=="normal") mixed=b;
        else if(mode=="screen") mixed=screen(a,b);
        else if(mode=="overlay"){
            mixed[0]=overlay1(a[0],b[0]);
            mixed[1]=overlay1(a[1],b[1]);
            mixed[2]=overlay1(a[2],b[2]);
        }
        else if(mode=="multiply") mixed={a[0]*b[0],a[1]*b[1],a[2]*b[2]};
        else if(mode=="add") mixed={clampf(a[0]+b[0]),clampf(a[1]+b[1]),clampf(a[2]+b[2])};
        else if(mode=="dodge") mixed={dodge1(a[0],b[0]),dodge1(a[1],b[1]),dodge1(a[2],b[2])};
        else mixed=b;
        r[0]=lerp(a[0],mixed[0],amount);
        r[1]=lerp(a[1],mixed[1],amount);
        r[2]=lerp(a[2],mixed[2],amount);
    }
    return to8u(result);
}

// =============================================================================
// SECTION 11 - KEYING
// =============================================================================

// --- 11.1 Chroma Key ---
static cv::Mat fx_chroma_key(const cv::Mat& f,cv::Vec3b key_color,
    float tol,float softness,const cv::Mat& bg){
    cv::Mat hsv; cv::cvtColor(f,hsv,cv::COLOR_BGR2HSV);
    cv::Mat key_hsv; cv::cvtColor(cv::Mat(1,1,CV_8UC3,key_color),key_hsv,cv::COLOR_BGR2HSV);
    float key_hue=key_hsv.at<cv::Vec3b>(0,0)[0];
    cv::Mat mask(f.rows,f.cols,CV_32F);
    for(int y=0;y<f.rows;y++)
    for(int x=0;x<f.cols;x++){
        auto& p=hsv.at<cv::Vec3b>(y,x);
        float dh=std::abs((float)p[0]-key_hue);
        if(dh>90) dh=180-dh;
        float ds=std::abs((float)p[1]/255.f-key_hsv.at<cv::Vec3b>(0,0)[1]/255.f);
        float key_strength=(dh/90.f+ds*0.5f)/std::max(0.01f,tol);
        mask.at<float>(y,x)=clampf(key_strength);
    }
    cv::GaussianBlur(mask,mask,cv::Size(0,0),softness);
    cv::Mat result=f.clone();
    if(!bg.empty()){
        cv::Mat bg_r; cv::resize(bg,bg_r,f.size());
        for(int y=0;y<f.rows;y++)
        for(int x=0;x<f.cols;x++){
            float m=mask.at<float>(y,x);
            auto& p=result.at<cv::Vec3b>(y,x);
            auto& q=bg_r.at<cv::Vec3b>(y,x);
            p[0]=(uchar)(p[0]*m+q[0]*(1-m));
            p[1]=(uchar)(p[1]*m+q[1]*(1-m));
            p[2]=(uchar)(p[2]*m+q[2]*(1-m));
        }
    }
    return result;
}

// --- 11.2 Luma Key ---
static cv::Mat fx_luma_key(const cv::Mat& f,float threshold,
    float tolerance,bool key_out_bright){
    cv::Mat result=f.clone();
    for(int y=0;y<f.rows;y++)
    for(int x=0;x<f.cols;x++){
        float lm=luma_bgr8(f.at<cv::Vec3b>(y,x));
        float diff=key_out_bright?(lm-threshold):(threshold-lm);
        float key=clampf(diff/std::max(0.01f,tolerance));
        auto& p=result.at<cv::Vec3b>(y,x);
        p[0]=(uchar)(p[0]*(1-key));
        p[1]=(uchar)(p[1]*(1-key));
        p[2]=(uchar)(p[2]*(1-key));
    }
    return result;
}

// =============================================================================
// SECTION 12 - CUSTOM AMV EFFECTS  (the signature ones)
// =============================================================================

// --- 12.1 Optical Flow Slomo (Twixtor equivalent) ---
// [Full implementation in pipeline section below]

// --- 12.2 RSMB Motion Blur ---
static cv::Mat fx_rsmb(const cv::Mat& cur,const cv::Mat& prev_gray_in,
    cv::Mat& prev_gray_out,float strength,float shutter,int samples){
    cv::Mat gray; cv::cvtColor(cur,gray,cv::COLOR_BGR2GRAY);
    prev_gray_out=gray.clone();
    if(prev_gray_in.empty()||prev_gray_in.size()!=gray.size()) return cur;
    float blur_frac=(shutter/360.f)*strength;
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prev_gray_in,gray,flow,0.5,3,12,3,5,1.2,0);
    std::vector<cv::Mat> fch; cv::split(flow,fch);
    int h=cur.rows,w=cur.cols;
    cv::Mat gx(h,w,CV_32F),gy(h,w,CV_32F);
    for(int y=0;y<h;y++)for(int x=0;x<w;x++){gx.at<float>(y,x)=(float)x;gy.at<float>(y,x)=(float)y;}
    cv::Mat acc=cv::Mat::zeros(h,w,CV_32FC3);
    for(int s=0;s<samples;s++){
        float t=(float)s/(samples-1)-0.5f;
        cv::Mat mx=gx+fch[0]*blur_frac*t;
        cv::Mat my=gy+fch[1]*blur_frac*t;
        cv::Mat warped; cv::remap(cur,warped,mx,my,cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        cv::Mat w32; warped.convertTo(w32,CV_32FC3);
        acc+=w32;
    }
    cv::Mat r; cv::Mat tmp=acc/samples; tmp.convertTo(r,CV_8UC3);
    return r;
}

// --- 12.3 Deep Glow (full quality) ---
static cv::Mat fx_deep_glow_v2(const cv::Mat& f,float threshold,
    float intensity,float radius,cv::Vec3f tint_bgr,float t){
    cv::Mat f32=to32f(f);
    // Extract bright
    cv::Mat bright(f32.size(),CV_32FC3);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto& p=f32.at<cv::Vec3f>(y,x);
        float lm=luma_bgr(p);
        float m=clampf((lm-threshold)/std::max(0.01f,1-threshold));
        m=m*m; // sharpen threshold
        bright.at<cv::Vec3f>(y,x)=p*m;
    }
    // 5-scale bloom
    float scales[]={0.8f,1.6f,3.2f,6.5f,13.f};
    float weights[]={0.35f,0.25f,0.20f,0.12f,0.08f};
    cv::Mat bloom=cv::Mat::zeros(f32.size(),CV_32FC3);
    int br=std::max(3,(int)(std::min(f32.cols,f32.rows)*0.01f*radius));
    for(int i=0;i<5;i++){
        int r=(int)(br*scales[i]); if(r%2==0)r++; if(r<3)r=3;
        cv::Mat bl; cv::GaussianBlur(bright,bl,cv::Size(r,r),r/3.0);
        cv::addWeighted(bloom,1,bl,weights[i],0,bloom);
    }
    // Tint
    for(int y=0;y<bloom.rows;y++)
    for(int x=0;x<bloom.cols;x++){
        auto& p=bloom.at<cv::Vec3f>(y,x);
        p[0]*=tint_bgr[0]; p[1]*=tint_bgr[1]; p[2]*=tint_bgr[2];
    }
    float pulse=0.92f+0.08f*(float)std::sin(2*(float)M_PI*0.8f*t+0.3f);
    bloom*=(intensity*pulse);
    // Screen blend
    cv::Mat result(f32.size(),CV_32FC3);
    for(int y=0;y<f32.rows;y++)
    for(int x=0;x<f32.cols;x++){
        auto b=f32.at<cv::Vec3f>(y,x);
        auto g=bloom.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        r[0]=clampf(1-(1-b[0])*(1-clampf(g[0])));
        r[1]=clampf(1-(1-b[1])*(1-clampf(g[1])));
        r[2]=clampf(1-(1-b[2])*(1-clampf(g[2])));
    }
    return to8u(result);
}

// --- 12.4 Datamosh V2 (improved) ---
static cv::Mat fx_datamosh_v2(const cv::Mat& f,const cv::Mat& prev,
    float intensity,int frame_idx,std::mt19937& rng){
    if(prev.empty()) return f;
    cv::Mat gc,gp; cv::cvtColor(f,gc,cv::COLOR_BGR2GRAY);
    cv::cvtColor(prev,gp,cv::COLOR_BGR2GRAY);
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(gp,gc,flow,0.5,2,8,2,5,1.1,0);
    std::vector<cv::Mat> fch; cv::split(flow,fch);
    int h=f.rows,w=f.cols;
    float scale=intensity*(6+std::sin(frame_idx*0.3f)*2);
    std::uniform_real_distribution<float> noise(-2,2);
    cv::Mat mx(h,w,CV_32F),my(h,w,CV_32F);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        mx.at<float>(y,x)=clamp((float)x+fch[0].at<float>(y,x)*scale+noise(rng),0.f,(float)(w-1));
        my.at<float>(y,x)=clamp((float)y+fch[1].at<float>(y,x)*scale+noise(rng),0.f,(float)(h-1));
    }
    cv::Mat dist; cv::remap(prev,dist,mx,my,cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    // Occasional full datamosh frame vs mixed
    float blend=(frame_idx%8<4)?intensity*0.9f:intensity*0.35f;
    cv::Mat result;
    cv::addWeighted(f,1-blend,dist,blend,0,result);
    // Add chromatic split for digital corruption look
    if(intensity>0.5f){
        int px=(int)(intensity*5);
        std::vector<cv::Mat> ch; cv::split(result,ch);
        cv::Mat r_sh=cv::Mat::zeros(h,w,CV_8U);
        if(px<w) ch[2](cv::Rect(0,0,w-px,h)).copyTo(r_sh(cv::Rect(px,0,w-px,h)));
        ch[2]=r_sh;
        cv::merge(ch,result);
    }
    return result;
}

// --- 12.5 VHS V2 ---
static cv::Mat fx_vhs_v2(const cv::Mat& f,float intensity,float t,std::mt19937& rng){
    cv::Mat result=f.clone();
    int h=f.rows,w=f.cols;
    std::uniform_real_distribution<float> ud(0,1);
    std::uniform_int_distribution<int> bd(0,(int)(intensity*80));
    // Random noise bands
    int n_bands=2+(int)(intensity*5);
    for(int i=0;i<n_bands;i++){
        int y=(int)(ud(rng)*h);
        int bh=2+(int)(ud(rng)*h*0.04f);
        for(int dy=0;dy<bh&&y+dy<h;dy++)
        for(int x=0;x<w;x++){
            auto& p=result.at<cv::Vec3b>(y+dy,x);
            int n=bd(rng);
            p[0]=clamp((int)p[0]+n,0,255);
            p[1]=clamp((int)p[1]+n,0,255);
            p[2]=clamp((int)p[2]+n,0,255);
        }
    }
    // Scanlines
    for(int y=0;y<h;y+=4){
        for(int x=0;x<w;x++){
            auto& p=result.at<cv::Vec3b>(y,x);
            float d=1-0.20f*intensity;
            p[0]=(uchar)(p[0]*d);p[1]=(uchar)(p[1]*d);p[2]=(uchar)(p[2]*d);
        }
    }
    // Chromatic shift (horizontal)
    int shift=(int)(intensity*6);
    if(shift>0&&shift<w){
        std::vector<cv::Mat> ch; cv::split(result,ch);
        cv::Mat r_sh=cv::Mat::zeros(h,w,CV_8U);
        ch[2](cv::Rect(0,0,w-shift,h)).copyTo(r_sh(cv::Rect(shift,0,w-shift,h)));
        ch[2]=r_sh;
        cv::merge(ch,result);
    }
    // Top/bottom degradation
    int deg=(int)(h*0.06f*intensity);
    for(int y=0;y<deg;y++){
        float a=(float)(deg-y)/deg*intensity;
        for(int x=0;x<w;x++){
            auto& p=result.at<cv::Vec3b>(y,x);
            int n=(int)(a*80*ud(rng));
            p[0]=clamp((int)p[0]+n,0,255);
            p[1]=clamp((int)p[1]+n,0,255);
        }
    }
    // Tracking instability (horizontal shift on some lines)
    int track_line=(int)(ud(rng)*h);
    int track_h=(int)(ud(rng)*8)+2;
    int track_shift=(int)((ud(rng)-0.5f)*intensity*30);
    for(int dy=0;dy<track_h&&track_line+dy<h;dy++){
        cv::Mat row=result.row(track_line+dy).clone();
        for(int x=0;x<w;x++){
            int sx=(x+track_shift+w)%w;
            result.at<cv::Vec3b>(track_line+dy,x)=row.at<cv::Vec3b>(0,sx);
        }
    }
    return result;
}

// --- 12.6 Chromatic Aberration V2 (radial) ---
static cv::Mat fx_chromatic_v2(const cv::Mat& f,float strength,float t,
    bool radial){
    int h=f.rows,w=f.cols;
    float px=strength*8*(0.7f+0.3f*(float)std::sin(2*(float)M_PI*3.5f*t));
    if(!radial){
        // Simple horizontal shift
        std::vector<cv::Mat> ch; cv::split(f,ch);
        int ipx=(int)std::abs(px);
        if(ipx>=w) return f;
        cv::Mat r_sh=cv::Mat::zeros(h,w,CV_8U);
        cv::Mat b_sh=cv::Mat::zeros(h,w,CV_8U);
        ch[2](cv::Rect(0,0,w-ipx,h)).copyTo(r_sh(cv::Rect(ipx,0,w-ipx,h)));
        ch[0](cv::Rect(ipx,0,w-ipx,h)).copyTo(b_sh(cv::Rect(0,0,w-ipx,h)));
        std::vector<cv::Mat> out={b_sh,ch[1],r_sh};
        cv::Mat result; cv::merge(out,result); return result;
    }
    // Radial: diverge from center
    float cx=w/2.f,cy=h/2.f;
    std::vector<cv::Mat> ch; cv::split(f,ch);
    std::vector<cv::Mat> out(3);
    for(int c=0;c<3;c++){
        float scale=1.f+(c==0?-0.008f:c==2?0.008f:0.f)*strength;
        cv::Mat M=(cv::Mat_<double>(2,3)<<scale,0,(1-scale)*cx,0,scale,(1-scale)*cy);
        cv::warpAffine(ch[c],out[c],M,cv::Size(w,h),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    }
    cv::Mat result; cv::merge(out,result); return result;
}

// --- 12.7 Energy Aura V2 ---
static cv::Mat fx_energy_aura_v2(const cv::Mat& f,float intensity,
    float t,cv::Vec3b color,int n_layers){
    cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
    cv::Mat blur; cv::GaussianBlur(gray,blur,cv::Size(0,0),1.5);
    cv::Mat edges; cv::Canny(blur,edges,55,145);
    cv::dilate(edges,edges,cv::Mat(),cv::Point(-1,-1),2);
    float pulse=0.7f+0.3f*(float)std::sin(2*(float)M_PI*2.8f*t+0.9f);
    float spark=0.5f+0.5f*(float)std::sin(2*(float)M_PI*4.6f*t+1.15f);
    cv::Mat glow=cv::Mat::zeros(f.rows,f.cols,CV_32FC3);
    float sigmas[]={1.5f,3.f,6.f,10.f,18.f};
    float ws[]={0.4f,0.25f,0.18f,0.1f,0.07f};
    for(int i=0;i<std::min(n_layers,5);i++){
        cv::Mat bl;
        int k=(int)(sigmas[i]*3)|1; if(k<3)k=3;
        cv::GaussianBlur(edges,bl,cv::Size(k,k),sigmas[i]);
        float w=ws[i]*intensity*pulse;
        for(int y=0;y<f.rows;y++)
        for(int x=0;x<f.cols;x++){
            float v=bl.at<uchar>(y,x)/255.f*w;
            auto& p=glow.at<cv::Vec3f>(y,x);
            p[0]+=v*(color[0]/255.f); p[1]+=v*(color[1]/255.f); p[2]+=v*(color[2]/255.f);
        }
    }
    // Inner spark core
    cv::Mat inner; cv::GaussianBlur(edges,inner,cv::Size(3,3),0.8);
    float sp=intensity*spark*0.5f;
    for(int y=0;y<f.rows;y++)
    for(int x=0;x<f.cols;x++){
        float v=inner.at<uchar>(y,x)/255.f*sp;
        auto& p=glow.at<cv::Vec3f>(y,x);
        p[0]=clampf(p[0]+v); p[1]=clampf(p[1]+v); p[2]=clampf(p[2]+v);
    }
    // White hot core
    cv::Mat core; cv::erode(edges,core,cv::Mat(),cv::Point(-1,-1),1);
    cv::GaussianBlur(core,core,cv::Size(3,3),0.5);
    float wc=intensity*spark*0.4f;
    for(int y=0;y<f.rows;y++)
    for(int x=0;x<f.cols;x++){
        float v=core.at<uchar>(y,x)/255.f*wc;
        auto& p=glow.at<cv::Vec3f>(y,x);
        p[0]=clampf(p[0]+v*0.6f); p[1]=clampf(p[1]+v*0.7f); p[2]=clampf(p[2]+v*0.9f);
    }
    // Screen blend
    cv::Mat f32=to32f(f);
    cv::Mat result(f.rows,f.cols,CV_32FC3);
    for(int y=0;y<f.rows;y++)
    for(int x=0;x<f.cols;x++){
        auto b=f32.at<cv::Vec3f>(y,x);
        auto g=glow.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        r[0]=clampf(1-(1-b[0])*(1-clampf(g[0])));
        r[1]=clampf(1-(1-b[1])*(1-clampf(g[1])));
        r[2]=clampf(1-(1-b[2])*(1-clampf(g[2])));
    }
    return to8u(result);
}

// --- 12.8 Fire Outline V2 ---
static cv::Mat fx_fire_outline_v2(const cv::Mat& f,float intensity,float t){
    cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
    cv::Mat smooth; cv::GaussianBlur(gray,smooth,cv::Size(5,5),1.2);
    cv::Mat edges; cv::Canny(smooth,edges,55,148);
    cv::dilate(edges,edges,cv::Mat(),cv::Point(-1,-1),1);
    float pulse=0.74f+0.26f*(float)std::sin(2*(float)M_PI*3.4f*t+0.6f);
    float spark=0.5f+0.5f*(float)std::sin(2*(float)M_PI*4.6f*t+1.15f);
    cv::Mat fire=cv::Mat::zeros(f.rows,f.cols,CV_32FC3);
    // Outer orange, mid yellow, inner white
    float sig[]={6.f,3.f,1.5f,0.7f};
    for(int i=0;i<4;i++){
        cv::Mat bl; int k=(int)(sig[i]*3)|1; if(k<3)k=3;
        cv::GaussianBlur(edges,bl,cv::Size(k,k),sig[i]);
        for(int y=0;y<f.rows;y++)
        for(int x=0;x<f.cols;x++){
            float v=bl.at<uchar>(y,x)/255.f;
            auto& p=fire.at<cv::Vec3f>(y,x);
            // i=0:outer red, i=1:orange, i=2:yellow, i=3:white
            float intensities[]={0.9f,0.65f,0.45f,0.90f};
            p[2]=clampf(p[2]+v*intensities[i]*intensity*pulse); // R
            if(i>=1) p[1]=clampf(p[1]+v*intensities[i]*0.5f*intensity*pulse); // G
            if(i>=2) p[0]=clampf(p[0]+v*intensities[i]*0.15f*intensity*spark); // B
            if(i==3){p[0]=clampf(p[0]+v*0.7f*intensity*spark);p[1]=clampf(p[1]+v*0.8f*intensity*spark);}
        }
    }
    float gs=0.22f+0.12f*intensity;
    cv::Mat f32=to32f(f);
    cv::Mat result(f.rows,f.cols,CV_32FC3);
    for(int y=0;y<f.rows;y++)
    for(int x=0;x<f.cols;x++){
        auto b=f32.at<cv::Vec3f>(y,x);
        auto g=fire.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        r[0]=clampf(b[0]+g[0]*gs); r[1]=clampf(b[1]+g[1]*gs); r[2]=clampf(b[2]+g[2]*gs);
    }
    return to8u(result);
}

// --- 12.9 Anamorphic Flare V2 ---
static cv::Mat fx_anamorphic_flare_v2(const cv::Mat& f,float intensity,float t){
    int h=f.rows,w=f.cols;
    cv::Mat f32=to32f(f);
    // Find brightest point
    cv::Mat luma(h,w,CV_32F);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) luma.at<float>(y,x)=luma_bgr(f32.at<cv::Vec3f>(y,x));
    cv::Mat bright; cv::threshold(luma,bright,0.80f,1,cv::THRESH_TOZERO);
    // Row-max for streak source
    cv::Mat streak_src(1,w,CV_32FC3,cv::Scalar(0,0,0));
    for(int x=0;x<w;x++){
        float mx=0; for(int y=0;y<h;y++) mx=std::max(mx,bright.at<float>(y,x));
        streak_src.at<cv::Vec3f>(0,x)={mx*0.5f,mx*0.65f,mx}; // blue-ish
    }
    cv::Mat streak; cv::resize(streak_src,streak,cv::Size(w,h),0,0,cv::INTER_LINEAR);
    // Blur massively horizontal
    int bw=w|1;
    cv::GaussianBlur(streak,streak,cv::Size(bw,3),w*0.04f,2.f);
    float pulse=0.8f+0.2f*(float)std::sin(2*(float)M_PI*0.75f*t+0.3f);
    float blend=(0.08f+0.10f*intensity)*pulse;
    cv::Mat result(h,w,CV_32FC3);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        auto b=f32.at<cv::Vec3f>(y,x);
        auto s=streak.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        r[0]=clampf(1-(1-b[0])*(1-s[0]*blend));
        r[1]=clampf(1-(1-b[1])*(1-s[1]*blend));
        r[2]=clampf(1-(1-b[2])*(1-s[2]*blend));
    }
    return to8u(result);
}

// --- 12.10 God Rays V2 ---
static cv::Mat fx_god_rays_v2(const cv::Mat& f,float intensity,float t,
    float cx_f=0.5f,float cy_f=0.1f){
    int h=f.rows,w=f.cols;
    cv::Mat f32=to32f(f);
    // Extract bright highlights
    cv::Mat luma(h,w,CV_32F);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++) luma.at<float>(y,x)=luma_bgr(f32.at<cv::Vec3f>(y,x));
    cv::Mat bright; cv::threshold(luma,bright,0.60f,1,cv::THRESH_TOZERO);
    // Radial streak from light source
    float lx=cx_f*w, ly=cy_f*h;
    cv::Mat rays=cv::Mat::zeros(h,w,CV_32F);
    int n_steps=16;
    float decay=0.93f;
    cv::Mat acc_r=bright.clone();
    for(int s=0;s<n_steps;s++){
        float scale=1.f+s*0.008f;
        float ox=(1-scale)*lx, oy=(1-scale)*ly;
        cv::Mat M=(cv::Mat_<double>(2,3)<<scale,0,ox,0,scale,oy);
        cv::Mat warped; cv::warpAffine(acc_r,warped,M,cv::Size(w,h),cv::INTER_LINEAR,cv::BORDER_CONSTANT);
        warped*=powf(decay,(float)s);
        rays+=warped;
    }
    rays/=n_steps;
    cv::normalize(rays,rays,0,1,cv::NORM_MINMAX);
    float blend=intensity*0.45f;
    cv::Mat result(h,w,CV_32FC3);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        auto b=f32.at<cv::Vec3f>(y,x);
        float r=rays.at<float>(y,x)*blend;
        auto& out=result.at<cv::Vec3f>(y,x);
        out[0]=clampf(b[0]+r*0.85f); out[1]=clampf(b[1]+r*0.9f); out[2]=clampf(b[2]+r);
    }
    return to8u(result);
}

// --- 12.11 Speed Lines V2 ---
static cv::Mat fx_speed_lines_v2(const cv::Mat& f,float density,
    float t,std::mt19937& rng,float cx_f=0.5f,float cy_f=0.5f){
    int h=f.rows,w=f.cols;
    cv::Mat overlay=cv::Mat::zeros(h,w,CV_8UC3);
    float cx=cx_f*w, cy=cy_f*h;
    float diag=std::sqrt((float)(w*w+h*h));
    int n=(int)(80+density*100);
    std::uniform_real_distribution<float> ang(0,2*(float)M_PI);
    std::uniform_real_distribution<float> ri(0.03f,0.12f);
    std::uniform_real_distribution<float> ro(0.4f,0.8f);
    std::uniform_int_distribution<int> lw_d(1,std::max(1,(int)(density*4)));
    std::uniform_int_distribution<int> al(80,230);
    for(int i=0;i<n;i++){
        float a=ang(rng);
        float r1=ri(rng)*std::min(w,h);
        float r2=ro(rng)*diag;
        int lw=lw_d(rng);
        int alpha=al(rng);
        cv::line(overlay,
            {(int)(cx+r1*std::cos(a)),(int)(cy+r1*std::sin(a))},
            {(int)(cx+r2*std::cos(a)),(int)(cy+r2*std::sin(a))},
            cv::Scalar((double)alpha,(double)alpha,(double)alpha),lw,cv::LINE_AA);
    }
    cv::GaussianBlur(overlay,overlay,cv::Size(3,3),0.8);
    // Screen blend
    cv::Mat f32=to32f(f); cv::Mat ov32=to32f(overlay);
    cv::Mat result(h,w,CV_32FC3);
    for(int y=0;y<h;y++) for(int x=0;x<w;x++){
        auto b=f32.at<cv::Vec3f>(y,x); auto g=ov32.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        r[0]=clampf(1-(1-b[0])*(1-g[0]));
        r[1]=clampf(1-(1-b[1])*(1-g[1]));
        r[2]=clampf(1-(1-b[2])*(1-g[2]));
    }
    return to8u(result);
}

// --- 12.12 Light Leak V2 ---
static cv::Mat fx_light_leak_v2(const cv::Mat& f,float intensity,
    float t,cv::Vec3f leak_bgr){
    int h=f.rows,w=f.cols;
    // Two moving off-screen light sources
    float side=(std::sin(0.42f*t)>=0)?-1.f:1.f;
    float cx1=(side<0?-0.18f:1.18f+0.05f*(float)std::sin(0.7f*t))*w;
    float cy1=(0.22f+0.28f*(0.5f+0.5f*(float)std::sin(0.34f*t+0.6f)))*h;
    float cx2=(side<0?1.16f:-0.16f+0.04f*(float)std::cos(0.63f*t+1.2f))*w;
    float cy2=(0.58f+0.26f*(0.5f+0.5f*(float)std::cos(0.29f*t+0.9f)))*h;
    float r1=0.72f*std::max(w,h), r2=0.66f*std::max(w,h);
    float pulse=0.8f+0.2f*(float)std::sin(2*(float)M_PI*0.75f*t+0.2f);
    cv::Mat f32=to32f(f);
    cv::Mat result(h,w,CV_32FC3);
    float blend=0.08f+0.04f*intensity;
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float d1=std::sqrt(powf(x-cx1,2)+powf(y-cy1,2));
        float d2=std::sqrt(powf(x-cx2,2)+powf(y-cy2,2));
        float m1=clampf(1-d1/r1), m2=clampf(1-d2/r2);
        float a=clampf((m1*0.72f+m2*0.55f)*pulse*intensity);
        auto b=f32.at<cv::Vec3f>(y,x);
        auto& r=result.at<cv::Vec3f>(y,x);
        cv::Vec3f leak={a*leak_bgr[0],a*leak_bgr[1],a*leak_bgr[2]};
        r[0]=clampf(1-(1-b[0])*(1-leak[0]*blend));
        r[1]=clampf(1-(1-b[1])*(1-leak[1]*blend));
        r[2]=clampf(1-(1-b[2])*(1-leak[2]*blend));
    }
    return to8u(result);
}

// --- 12.13 Vignette V2 (animated) ---
static cv::Mat fx_vignette_v2(const cv::Mat& f,float strength,float t,bool animated){
    float s=strength;
    if(animated) s*=(0.92f+0.08f*(float)std::sin(2*(float)M_PI*0.5f*t));
    int h=f.rows,w=f.cols;
    cv::Mat f32=to32f(f);
    for(int y=0;y<h;y++)
    for(int x=0;x<w;x++){
        float nx=(float)x/(w-1)*2-1, ny=(float)y/(h-1)*2-1;
        float d=std::sqrt(nx*nx+ny*ny);
        float mask=1-clampf((d-0.2f)/0.8f)*s;
        auto& p=f32.at<cv::Vec3f>(y,x);
        p[0]*=mask;p[1]*=mask;p[2]*=mask;
    }
    return to8u(f32);
}

// --- 12.14 Twitch V2 ---
struct TwitchV2 {
    std::mt19937 rng;
    std::vector<float> beats;
    float active_until=-1;
    bool active=false;
    bool do_shake,do_blur,do_color,do_bright,do_hue;
    float str=0;

    void init(float bpm,float dur){
        float bi=60.f/bpm;
        for(float t2=0;t2<dur;t2+=bi) beats.push_back(t2);
    }
    void update(float t,float intensity){
        active=false;
        for(float bt:beats){
            if(std::abs(t-bt)<0.12f){
                active=true; active_until=bt+0.18f;
                std::uniform_int_distribution<int> op(1,4);
                int n=op(rng);
                do_shake=do_blur=do_color=do_bright=do_hue=false;
                for(int i=0;i<n;i++){
                    std::uniform_int_distribution<int> w(0,4);
                    switch(w(rng)){
                        case 0:do_shake=true;break;
                        case 1:do_blur=true;break;
                        case 2:do_color=true;break;
                        case 3:do_bright=true;break;
                        case 4:do_hue=true;break;
                    }
                }
                std::uniform_real_distribution<float> sd(0.4f,1.0f);
                str=sd(rng)*intensity;
                break;
            }
        }
        if(t>active_until) active=false;
    }
    cv::Mat apply(const cv::Mat& f,float t,float intensity){
        update(t,intensity);
        if(!active) return f;
        cv::Mat r=f.clone();
        if(do_shake){
            std::uniform_int_distribution<int> sd(-1,1);
            int dx=(int)(str*14)*sd(rng);
            int dy=(int)(str*10)*sd(rng);
            cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,dx,0,1,dy);
            cv::warpAffine(r,r,M,r.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        }
        if(do_blur){
            int k=(int)(str*10)|1; if(k<3)k=3;
            cv::GaussianBlur(r,r,cv::Size(k,k),str*4);
        }
        if(do_color){
            std::uniform_real_distribution<float> cd(0.7f,1.3f);
            float cr=cd(rng),cg=cd(rng),cb=cd(rng);
            std::vector<cv::Mat> ch; cv::split(r,ch);
            ch[2].convertTo(ch[2],-1,cr,0);
            ch[1].convertTo(ch[1],-1,cg,0);
            ch[0].convertTo(ch[0],-1,cb,0);
            cv::merge(ch,r);
        }
        if(do_bright) r.convertTo(r,-1,1+str*0.6f,0);
        if(do_hue){
            cv::Mat hsv; cv::cvtColor(r,hsv,cv::COLOR_BGR2HSV);
            std::uniform_int_distribution<int> hd(0,40);
            int hs=hd(rng);
            for(int y=0;y<hsv.rows;y++)for(int x=0;x<hsv.cols;x++){
                auto& p=hsv.at<cv::Vec3b>(y,x);
                p[0]=(uchar)((p[0]+hs)%180);
                p[1]=clamp((int)(p[1]*(1+str*0.3f)),0,255);
            }
            cv::cvtColor(hsv,r,cv::COLOR_HSV2BGR);
        }
        return r;
    }
};

// =============================================================================
// SECTION 13 - OPTICAL FLOW SLOMO (standalone pipeline)
// =============================================================================

static bool run_optical_flow_slomo(
    const std::string& in,const std::string& out,
    float speed,const std::string& quality,int crf)
{
    VInfo vi=probe(in);
    if(vi.total<2) return false;
    LOG("Optical flow slomo: speed="+std::to_string(speed));
    auto tmp=mktmp("of_");
    fs::path fdir=tmp/"frames"; fs::create_directories(fdir);
    // Extract
    std::ostringstream ex;
    ex<<" -y -i \""<<in<<"\""
      <<" -vf \"scale="<<vi.w<<":"<<vi.h<<"\""
      <<" -r "<<vi.fps
      <<" \""<<(fdir/"frame_%06d.png").string()<<"\"";
    if(!RUN_FFMPEG(ex.str(),120)){rmtmp(tmp);return false;}
    // Load
    std::vector<cv::Mat> src;
    for(int i=1;;i++){
        char b[32]; std::snprintf(b,sizeof(b),"frame_%06d.png",i);
        if(!fs::exists(fdir/b)) break;
        cv::Mat f=cv::imread((fdir/b).string());
        if(f.empty()) break;
        src.push_back(f);
    }
    if(src.size()<2){rmtmp(tmp);return false;}
    // Flow quality
    int pyr=(quality=="high")?4:(quality=="fast")?2:3;
    int ws=(quality=="high")?15:(quality=="fast")?8:12;
    int it=(quality=="high")?5:(quality=="fast")?2:3;
    int fbetween=std::max(1,(int)std::round(1.f/speed)-1);
    LOG("  "+std::to_string(fbetween)+" interp frames between each pair");
    fs::path odir=tmp/"out"; fs::create_directories(odir);
    int oidx=0;
    char fname[24]; std::snprintf(fname,sizeof(fname),"f_%06d.png",oidx++);
    cv::imwrite((odir/fname).string(),src[0]);
    int h=src[0].rows,w=src[0].cols;
    cv::Mat gx(h,w,CV_32F),gy(h,w,CV_32F);
    for(int y2=0;y2<h;y2++) for(int x2=0;x2<w;x2++){gx.at<float>(y2,x2)=(float)x2;gy.at<float>(y2,x2)=(float)y2;}
    for(size_t i=0;i+1<src.size();i++){
        cv::Mat ga,gb;
        cv::cvtColor(src[i],ga,cv::COLOR_BGR2GRAY);
        cv::cvtColor(src[i+1],gb,cv::COLOR_BGR2GRAY);
        cv::Mat fwd,bwd;
        try{
            cv::calcOpticalFlowFarneback(ga,gb,fwd,0.5,pyr,ws,it,5,1.2,0);
            cv::calcOpticalFlowFarneback(gb,ga,bwd,0.5,pyr,ws,it,5,1.2,0);
        }catch(...){
            for(int j=1;j<=fbetween;j++){
                float t=(float)j/(fbetween+1);
                cv::Mat bl; cv::addWeighted(src[i],1-t,src[i+1],t,0,bl);
                std::snprintf(fname,sizeof(fname),"f_%06d.png",oidx++);
                cv::imwrite((odir/fname).string(),bl);
            }
            std::snprintf(fname,sizeof(fname),"f_%06d.png",oidx++);
            cv::imwrite((odir/fname).string(),src[i+1]);
            continue;
        }
        std::vector<cv::Mat> ff,bf; cv::split(fwd,ff); cv::split(bwd,bf);
        for(int j=1;j<=fbetween;j++){
            float t2=(float)j/(fbetween+1);
            cv::Mat mx_a=gx+ff[0]*t2, my_a=gy+ff[1]*t2;
            cv::Mat mx_b=gx+bf[0]*(1-t2), my_b=gy+bf[1]*(1-t2);
            cv::Mat wa,wb;
            cv::remap(src[i],wa,mx_a,my_a,cv::INTER_LINEAR,cv::BORDER_REPLICATE);
            cv::remap(src[i+1],wb,mx_b,my_b,cv::INTER_LINEAR,cv::BORDER_REPLICATE);
            cv::Mat interp; cv::addWeighted(wa,1-t2,wb,t2,0,interp);
            std::snprintf(fname,sizeof(fname),"f_%06d.png",oidx++);
            cv::imwrite((odir/fname).string(),interp);
        }
        std::snprintf(fname,sizeof(fname),"f_%06d.png",oidx++);
        cv::imwrite((odir/fname).string(),src[i+1]);
        if((int)i%20==0) LOG("  Interpolating pair "+std::to_string(i)+"/"+std::to_string(src.size()));
    }
    bool ok=encode_frames(odir,out,vi.fps,vi.audio?in:"",crf);
    rmtmp(tmp);
    return ok;
}

// =============================================================================
// SECTION 14 - CONFIG & CLI
// =============================================================================

struct Cfg {
    std::string effect,preset,input,output;
    // Effect params
    float intensity=0.80f, strength=0.75f, speed=0.30f;
    float radius=0.70f, threshold=0.65f, amount=30.f;
    float sigma=8.f, scale=100.f, complexity=4.f;
    float opacity=0.50f, evolution=0.f, phase=0.f;
    float cx=0.5f, cy=0.42f, angle=0.f;
    float bpm=130.f, decay=0.65f;
    int   pixels=6, frames=5, spacing=4, thickness=2, levels=6;
    int   block_w=16, block_h=16, seed=42, n_layers=5;
    bool  invert=false, radial=true, animated=true;
    std::string mood="hype", color="warm", quality="medium";
    std::string blend_mode="screen", type="sparks";
    int crf=16;
    bool list=false;
};

static cv::Vec3b color_str(const std::string& s){
    if(s=="purple")  return {200,80,200};
    if(s=="cyan")    return {255,220,80};
    if(s=="blue")    return {255,100,40};
    if(s=="orange")  return {40,140,255};
    if(s=="red")     return {40,40,255};
    if(s=="green")   return {40,255,80};
    if(s=="yellow")  return {40,255,255};
    if(s=="pink")    return {200,80,255};
    if(s=="white")   return {255,255,255};
    if(s=="warm")    return {100,170,255};
    if(s=="cool")    return {255,180,100};
    return {255,200,100};
}

static cv::Vec3f color_str_f(const std::string& s){
    auto c=color_str(s);
    return {c[0]/255.f,c[1]/255.f,c[2]/255.f};
}

static Cfg parse(int argc,char* argv[]){
    Cfg c;
    for(int i=1;i<argc;i++){
        std::string k=argv[i];
        auto nx=[&]()->std::string{return(i+1<argc)?std::string(argv[++i]):"";};
        if(k=="--effect")    c.effect=nx();
        else if(k=="--preset") c.preset=nx();
        else if(k=="--input")  c.input=nx();
        else if(k=="--output") c.output=nx();
        else if(k=="--intensity") c.intensity=std::stof(nx());
        else if(k=="--strength")  c.strength=std::stof(nx());
        else if(k=="--speed")     c.speed=std::stof(nx());
        else if(k=="--radius")    c.radius=std::stof(nx());
        else if(k=="--threshold") c.threshold=std::stof(nx());
        else if(k=="--amount")    c.amount=std::stof(nx());
        else if(k=="--sigma")     c.sigma=std::stof(nx());
        else if(k=="--scale")     c.scale=std::stof(nx());
        else if(k=="--complexity") c.complexity=std::stof(nx());
        else if(k=="--opacity")   c.opacity=std::stof(nx());
        else if(k=="--evolution") c.evolution=std::stof(nx());
        else if(k=="--phase")     c.phase=std::stof(nx());
        else if(k=="--cx") c.cx=std::stof(nx());
        else if(k=="--cy") c.cy=std::stof(nx());
        else if(k=="--angle") c.angle=std::stof(nx());
        else if(k=="--bpm")   c.bpm=std::stof(nx());
        else if(k=="--decay") c.decay=std::stof(nx());
        else if(k=="--pixels")  c.pixels=std::stoi(nx());
        else if(k=="--frames")  c.frames=std::stoi(nx());
        else if(k=="--spacing") c.spacing=std::stoi(nx());
        else if(k=="--thickness") c.thickness=std::stoi(nx());
        else if(k=="--levels")  c.levels=std::stoi(nx());
        else if(k=="--block-w") c.block_w=std::stoi(nx());
        else if(k=="--block-h") c.block_h=std::stoi(nx());
        else if(k=="--seed")    c.seed=std::stoi(nx());
        else if(k=="--n-layers") c.n_layers=std::stoi(nx());
        else if(k=="--invert")  c.invert=true;
        else if(k=="--no-radial") c.radial=false;
        else if(k=="--no-animated") c.animated=false;
        else if(k=="--mood")   c.mood=nx();
        else if(k=="--color")  c.color=nx();
        else if(k=="--quality") c.quality=nx();
        else if(k=="--blend")  c.blend_mode=nx();
        else if(k=="--type")   c.type=nx();
        else if(k=="--crf")    c.crf=std::stoi(nx());
        else if(k=="--list")   c.list=true;
    }
    return c;
}

// =============================================================================
// SECTION 15 - PRESET SYSTEM
// =============================================================================

struct PresetStep { std::string fx; float i; std::string extra; };

static std::map<std::string,std::vector<PresetStep>> PRESETS={
    {"hype_phonk",{
        {"color_grade",0.88f,"hype"},{"deep_glow",0.78f,"warm"},
        {"chromatic",0.65f,""},{"vignette",0.60f,""},
        {"film_grain",0.48f,""},{"anamorphic_flare",0.65f,""},
    }},
    {"dark_villain",{
        {"color_grade",0.88f,"dark"},{"energy_aura",0.82f,"purple"},
        {"fire_outline",0.72f,""},{"chromatic",0.70f,""},
        {"scanlines",0.60f,""},{"vignette",0.78f,""},{"film_grain",0.52f,""},
    }},
    {"anime_cinematic",{
        {"color_grade",0.82f,"dramatic"},{"deep_glow",0.72f,"warm"},
        {"cartoon",0.60f,""},{"anamorphic_flare",0.58f,""},
        {"vignette",0.65f,""},{"film_grain",0.42f,""},
    }},
    {"sad_emotional",{
        {"color_grade",0.82f,"sad"},{"deep_glow",0.58f,"cool"},
        {"vignette",0.72f,""},{"film_grain",0.62f,""},
        {"light_leak",0.48f,"sad"},{"photo_filter",0.35f,""},
    }},
    {"cyberpunk_city",{
        {"color_grade",0.88f,"cyberpunk"},{"energy_aura",0.82f,"cyan"},
        {"deep_glow",0.82f,"cyan"},{"chromatic",0.78f,""},
        {"anamorphic_flare",0.72f,""},{"scanlines",0.42f,""},{"film_grain",0.52f,""},
    }},
    {"retro_vhs",{
        {"vhs",0.88f,""},{"scanlines",0.62f,""},{"chromatic",0.68f,""},
        {"film_grain",0.72f,""},{"light_leak",0.52f,"warm"},{"vignette",0.58f,""},
    }},
    {"action_hype",{
        {"color_grade",0.90f,"hype"},{"speed_lines",0.82f,""},
        {"deep_glow",0.72f,"orange"},{"energy_aura",0.68f,"orange"},
        {"vignette",0.62f,""},{"film_grain",0.58f,""},
    }},
    {"midnight_neon",{
        {"color_grade",0.88f,"dark"},{"deep_glow",0.88f,"purple"},
        {"energy_aura",0.82f,"purple"},{"chromatic",0.72f,""},
        {"vignette",0.82f,""},{"film_grain",0.58f,""},{"scanlines",0.38f,""},
    }},
    {"impact_frame",{
        {"cartoon",0.65f,""},{"find_edges",0.50f,""},
        {"posterize",0.70f,""},{"color_grade",0.80f,"hype"},
    }},
    {"film_master",{
        {"color_grade",0.78f,"warm"},{"film_grain",0.72f,""},
        {"dust_scratches",0.65f,""},{"vignette",0.70f,""},
        {"anamorphic_flare",0.45f,""},{"deep_glow",0.42f,"warm"},
    }},
    {"data_glitch",{
        {"datamosh",0.85f,""},{"chromatic",0.80f,""},
        {"vhs",0.60f,""},{"scanlines",0.50f,""},{"color_grade",0.70f,"dark"},
    }},
    {"god_of_war",{
        {"color_grade",0.85f,"hype"},{"fire_outline",0.80f,""},
        {"energy_aura",0.75f,"orange"},{"god_rays",0.70f,""},
        {"deep_glow",0.68f,"orange"},{"vignette",0.65f,""},
        {"film_grain",0.50f,""},
    }},
};

// =============================================================================
// SECTION 16 - MAIN DISPATCHER
// =============================================================================

static bool run_single(Cfg& c){
    LOG("Effect: "+c.effect+" | "+c.input+" -> "+c.output);
    VInfo vi=probe(c.input);
    std::mt19937 rng(c.seed);
    LUT3D lut;

    // Special cases that need their own pipeline
    if(c.effect=="optical_flow_slomo"||c.effect=="slomo")
        return run_optical_flow_slomo(c.input,c.output,c.speed,c.quality,c.crf);

    if(c.effect=="motion_blur"||c.effect=="rsmb"){
        cv::Mat pg;
        return pipeline(c.input,c.output,[&](const cv::Mat& f,float t,int idx)->cv::Mat{
            cv::Mat result=fx_rsmb(f,pg,pg,c.strength,180,6);
            return result;
        },c.crf);
    }

    if(c.effect=="echo_trail"||c.effect=="echo"){
        std::deque<cv::Mat> buf;
        return pipeline(c.input,c.output,[&](const cv::Mat& f,float t,int idx)->cv::Mat{
            return fx_echo_trail_v2(f,buf,c.frames,c.decay,c.blend_mode);
        },c.crf);
    }

    if(c.effect=="datamosh"){
        cv::Mat prev;
        int fidx=0;
        return pipeline(c.input,c.output,[&](const cv::Mat& f,float t,int idx)->cv::Mat{
            auto r=fx_datamosh_v2(f,prev,c.intensity,fidx++,rng);
            prev=f.clone(); return r;
        },c.crf);
    }

    if(c.effect=="particles"){
        std::vector<Particle> parts;
        float grav=(c.type=="rain")?0.3f:(c.type=="sparks")?0.15f:(-0.02f);
        return pipeline(c.input,c.output,[&](const cv::Mat& f,float t,int idx)->cv::Mat{
            return fx_particles(f,t,parts,grav,0.f,c.type);
        },c.crf);
    }

    if(c.effect=="twitch"){
        TwitchV2 tw; tw.rng.seed(c.seed); tw.init(c.bpm,vi.dur);
        return pipeline(c.input,c.output,[&](const cv::Mat& f,float t,int idx)->cv::Mat{
            return tw.apply(f,t,c.intensity);
        },c.crf);
    }

    if(c.effect=="posterize_time"){
        posterize_frame_idx=0; posterize_last=cv::Mat();
        float tfps=c.scale>0?c.scale:12.f;
        return pipeline(c.input,c.output,[&](const cv::Mat& f,float t,int idx)->cv::Mat{
            return fx_posterize_time(f,(float)vi.fps,tfps);
        },c.crf);
    }

    if(c.effect=="shatter"){
        return pipeline(c.input,c.output,[&](const cv::Mat& f,float t,int idx)->cv::Mat{
            float dur=(float)vi.dur;
            float prog=std::min(1.f,t/std::max(0.5f,dur));
            return fx_shatter(f,prog,(int)(c.scale),c.seed);
        },c.crf);
    }

    if(c.effect=="color_grade"){
        lut=build_lut(c.mood,c.intensity);
        return pipeline(c.input,c.output,[&](const cv::Mat& f,float t,int idx)->cv::Mat{
            cv::Mat r=f.clone(); lut.apply(r); return r;
        },c.crf);
    }

    // All other per-frame effects
    return pipeline(c.input,c.output,[&](const cv::Mat& f,float t,int idx)->cv::Mat{
        auto& e=c.effect;

        // --- Blur & Sharpen ---
        if(e=="gaussian_blur")   return fx_gaussian_blur(f,c.sigma);
        if(e=="directional_blur") return fx_directional_blur(f,std::max(3,(int)c.amount),c.angle);
        if(e=="radial_blur")     return fx_radial_blur(f,c.amount/100.f,c.cx,c.cy);
        if(e=="bilateral_blur")  return fx_bilateral_blur(f,9,c.sigma*10,c.sigma*10);
        if(e=="box_blur")        return fx_box_blur(f,std::max(1,(int)c.sigma));
        if(e=="camera_lens_blur") return fx_camera_lens_blur(f,std::max(3,(int)c.sigma),6);
        if(e=="smart_blur")      return fx_smart_blur(f,c.sigma,c.amount);
        if(e=="unsharp_mask")    return fx_unsharp_mask(f,c.intensity,c.sigma);
        if(e=="channel_blur")    return fx_channel_blur(f,c.sigma,c.sigma*0.8f,c.sigma*0.6f);
        if(e=="compound_blur")   return fx_compound_blur(f,c.sigma*2);
        if(e=="sharpen")         return fx_unsharp_mask(f,c.strength,1.5f);

        // --- Color Correction ---
        if(e=="levels")    return fx_levels(f,0,1,1.f/std::max(0.1f,c.intensity),0,1);
        if(e=="hue_sat")   return fx_hue_sat_light(f,c.angle,c.intensity,c.opacity);
        if(e=="vibrance")  return fx_vibrance(f,c.intensity,c.strength-0.5f);
        if(e=="photo_filter") return fx_photo_filter(f,color_str_f(c.color),c.intensity,true);
        if(e=="tint")      return fx_tint(f,{0,0,0},{1,1,0.8f},c.intensity);
        if(e=="shadows_highlights") return fx_shadows_highlights(f,c.intensity*0.3f,c.intensity*0.2f,c.strength-0.5f);
        if(e=="invert")    return fx_invert(f,true,true,true);
        if(e=="color_balance") return fx_color_balance(f,-0.02f,-0.01f,0.05f,0,0,0,0.05f,0.02f,-0.04f);
        if(e=="three_way_color"){
            cv::Vec3f lift={-0.02f,-0.01f,0.03f},gamma={0,0,0},gain={0.02f,0.01f,-0.02f};
            return fx_three_way_color(f,lift,gamma,gain);
        }

        // --- Distort ---
        if(e=="turbulent_displace") return fx_turbulent_displace(f,c.amount,c.scale,c.evolution+t,c.seed);
        if(e=="wave_warp")    return fx_wave_warp(f,c.amount,c.scale,c.phase+t*2,c.angle);
        if(e=="ripple")       return fx_ripple(f,c.scale,c.amount,c.phase+t*3,c.cx,c.cy);
        if(e=="bulge")        return fx_bulge(f,c.intensity,c.radius,c.cx,c.cy);
        if(e=="twirl")        return fx_twirl(f,c.angle,c.radius,c.cx,c.cy);
        if(e=="spherize")     return fx_spherize(f,c.intensity,c.cx,c.cy);
        if(e=="mirror")       return fx_mirror(f,c.angle,c.cx,c.cy);
        if(e=="polar_coords") return fx_polar_coordinates(f,c.intensity,true);
        if(e=="liquify")      return fx_liquify(f,c.cx,c.cy,c.radius,c.intensity,t);

        // --- Generate ---
        if(e=="lightning"){
            cv::Point2f s={c.cx*(float)vi.w,0};
            cv::Point2f en={c.cx*(float)vi.w,(float)vi.h};
            return fx_advanced_lightning(f,s,en,(int)(c.intensity*4)+1,
                color_str(c.color),c.intensity,t);
        }
        if(e=="lens_flare")    return fx_lens_flare(f,c.cx,c.cy,c.intensity,c.scale/100.f);
        if(e=="fractal_noise") return fx_fractal_noise(f,c.scale,c.complexity,c.evolution+t,c.opacity,c.seed);
        if(e=="cell_pattern")  return fx_cell_pattern(f,c.scale,c.sigma,c.evolution+t,c.opacity,c.seed);
        if(e=="caustics")      return fx_caustics(f,t,c.intensity);
        if(e=="god_rays")      return fx_god_rays_v2(f,c.intensity,t,c.cx,0.1f);

        // --- Stylize ---
        if(e=="glow")          return fx_glow(f,c.threshold,c.radius,c.intensity);
        if(e=="find_edges")    return fx_find_edges(f,c.intensity,c.invert);
        if(e=="cartoon")       return fx_cartoon(f,(int)(c.sigma),c.amount,c.levels,c.intensity);
        if(e=="emboss")        return fx_emboss(f,c.angle,c.intensity,c.strength);
        if(e=="mosaic")        return fx_mosaic(f,c.block_w,c.block_h);
        if(e=="posterize")     return fx_posterize(f,c.levels);
        if(e=="scatter")       return fx_scatter(f,(int)c.amount,c.seed+(int)(t*100));
        if(e=="roughen_edges") return fx_roughen_edges(f,c.amount,t,c.seed);
        if(e=="strobe_light")  return fx_strobe_light(f,t,c.bpm/60.f,c.intensity);
        if(e=="motion_tile")   return fx_motion_tile(f,0,t*0.5f,0,0);
        if(e=="threshold"){
            cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
            cv::Mat r; cv::threshold(gray,r,c.threshold*255,255,cv::THRESH_BINARY);
            cv::cvtColor(r,r,cv::COLOR_GRAY2BGR); return r;
        }

        // --- Noise & Grain ---
        if(e=="film_grain")     return fx_film_grain_v2(f,c.strength,t,c.seed);
        if(e=="dust_scratches") return fx_dust_scratches(f,c.intensity,c.sigma,c.seed);
        if(e=="median")         return fx_median(f,(int)(c.sigma)|1);

        // --- Time ---
        if(e=="time_displacement"){
            cv::Mat map=cv::Mat::zeros(f.rows,f.cols,CV_8UC3);
            return fx_time_displacement(f,f,map,c.amount);
        }

        // --- Perspective ---
        if(e=="drop_shadow")  return fx_drop_shadow(f,(int)(c.cx*20),(int)(c.cy*20),c.sigma,c.intensity,{0,0,0});
        if(e=="bevel_edges")  return fx_bevel_edges(f,(int)(c.sigma),c.angle);

        // --- Channel ---
        if(e=="shift_channels") return fx_shift_channels(f,c.pixels,0,-c.pixels);
        if(e=="minimax_dilate") return fx_minimax(f,(int)c.sigma,true);
        if(e=="minimax_erode")  return fx_minimax(f,(int)c.sigma,false);
        if(e=="invert_channel") return fx_invert(f,true,true,true);

        // --- Keying ---
        if(e=="chroma_key")  return fx_chroma_key(f,{0,180,0},c.threshold,c.sigma,cv::Mat());
        if(e=="luma_key")    return fx_luma_key(f,c.threshold,c.sigma,c.invert);

        // --- Custom AMV ---
        if(e=="deep_glow")       return fx_deep_glow_v2(f,c.threshold,c.intensity,c.radius,color_str_f(c.color),t);
        if(e=="chromatic"||e=="chromatic_aberration") return fx_chromatic_v2(f,c.intensity,t,c.radial);
        if(e=="energy_aura")     return fx_energy_aura_v2(f,c.intensity,t,color_str(c.color),c.n_layers);
        if(e=="fire_outline")    return fx_fire_outline_v2(f,c.intensity,t);
        if(e=="anamorphic_flare")return fx_anamorphic_flare_v2(f,c.intensity,t);
        if(e=="speed_lines")     return fx_speed_lines_v2(f,c.intensity,t,rng,c.cx,c.cy);
        if(e=="light_leak")      return fx_light_leak_v2(f,c.intensity,t,color_str_f(c.color));
        if(e=="vignette")        return fx_vignette_v2(f,c.strength,t,c.animated);
        if(e=="vhs")             return fx_vhs_v2(f,c.intensity,t,rng);
        if(e=="ink_outline"){
            return fx_cartoon(f,5,c.amount>0?c.amount:60,255,c.intensity);
        }
        if(e=="scanlines")       return cv::Mat([&]()->cv::Mat{
            cv::Mat r=f.clone();
            for(int y=0;y<r.rows;y+=c.spacing){
                for(int x=0;x<r.cols;x++){
                    auto& p=r.at<cv::Vec3b>(y,x);
                    float d=1-0.22f*c.intensity;
                    p[0]=(uchar)(p[0]*d);p[1]=(uchar)(p[1]*d);p[2]=(uchar)(p[2]*d);
                }
            }
            return r;
        }());
        if(e=="color_dodge_aura"){
            cv::Mat gray; cv::cvtColor(f,gray,cv::COLOR_BGR2GRAY);
            cv::Mat edges; cv::Canny(gray,edges,50,130);
            cv::Mat glow_src; cv::GaussianBlur(edges,glow_src,cv::Size(0,0),8);
            float pulse=0.75f+0.25f*(float)std::sin(2*(float)M_PI*3.2f*t+0.5f);
            cv::Vec3f dc=color_str_f(c.color);
            cv::Mat f32=to32f(f);
            cv::Mat glow=cv::Mat::zeros(f.rows,f.cols,CV_32FC3);
            for(int y=0;y<f.rows;y++) for(int x=0;x<f.cols;x++){
                float v=glow_src.at<uchar>(y,x)/255.f*c.intensity*pulse;
                glow.at<cv::Vec3f>(y,x)={v*dc[0],v*dc[1],v*dc[2]};
            }
            cv::Mat result(f.rows,f.cols,CV_32FC3);
            for(int y=0;y<f.rows;y++) for(int x=0;x<f.cols;x++){
                auto b=f32.at<cv::Vec3f>(y,x); auto g=glow.at<cv::Vec3f>(y,x);
                auto& r=result.at<cv::Vec3f>(y,x);
                r[0]=lerp(b[0],clampf(b[0]/std::max(1e-4f,1-g[0])),c.intensity*0.5f);
                r[1]=lerp(b[1],clampf(b[1]/std::max(1e-4f,1-g[1])),c.intensity*0.5f);
                r[2]=lerp(b[2],clampf(b[2]/std::max(1e-4f,1-g[2])),c.intensity*0.5f);
            }
            return to8u(result);
        }
        if(e=="shatter"){
            float dur=(float)vi.dur;
            float prog=std::min(1.f,t/std::max(0.5f,dur));
            return fx_shatter(f,prog,100,c.seed);
        }

        LOG("Unknown effect: "+e);
        return f;
    },c.crf);
}

static bool run_preset(Cfg& c){
    auto it=PRESETS.find(c.preset);
    if(it==PRESETS.end()){
        LOG("Unknown preset: "+c.preset);
        LOG("Available:");
        for(auto& kv:PRESETS) LOG("  "+kv.first);
        return false;
    }
    const auto& steps=it->second;
    LOG("Preset '"+c.preset+"': "+std::to_string(steps.size())+" effects");
    std::string cur=c.input;
    std::string final=c.output;
    std::vector<std::string> tmps;
    for(size_t i=0;i<steps.size();i++){
        bool last=(i==steps.size()-1);
        std::string out;
        if(last){ out=final; }
        else{
            out=(fs::temp_directory_path()/(
                "aue_step_"+std::to_string(i)+"_"+
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".mp4"
            )).string();
            tmps.push_back(out);
        }
        Cfg sc=c;
        sc.effect=steps[i].fx;
        sc.intensity=sc.strength=steps[i].i;
        if(!steps[i].extra.empty()){sc.mood=steps[i].extra;sc.color=steps[i].extra;}
        sc.input=cur; sc.output=out;
        LOG("  Step "+std::to_string(i+1)+"/"+std::to_string(steps.size())+": "+sc.effect);
        if(!run_single(sc)){
            LOG("  Step failed, passing through");
            if(!last) fs::copy_file(cur,out,fs::copy_options::overwrite_existing);
        }
        cur=out;
    }
    for(auto& f:tmps) try{fs::remove(f);}catch(...){}
    return fs::exists(final);
}

static void print_all(){
    std::cout<<"ARCHIVE ULTIMATE EFFECTS ENGINE v"<<EXE_VERSION<<"\n\n";
    std::cout<<"BLUR & SHARPEN:\n";
    for(auto& e:{"gaussian_blur","directional_blur","radial_blur","bilateral_blur",
                  "box_blur","camera_lens_blur","smart_blur","unsharp_mask",
                  "channel_blur","compound_blur","sharpen"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"COLOR CORRECTION:\n";
    for(auto& e:{"color_grade","levels","hue_sat","vibrance","photo_filter","tint",
                  "shadows_highlights","invert","color_balance","three_way_color"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"DISTORT:\n";
    for(auto& e:{"turbulent_displace","wave_warp","ripple","bulge","twirl",
                  "spherize","mirror","polar_coords","liquify"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"GENERATE:\n";
    for(auto& e:{"lightning","lens_flare","fractal_noise","cell_pattern","caustics","god_rays"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"STYLIZE:\n";
    for(auto& e:{"glow","find_edges","cartoon","emboss","mosaic","posterize",
                  "scatter","roughen_edges","strobe_light","motion_tile","threshold"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"NOISE & GRAIN:\n";
    for(auto& e:{"film_grain","dust_scratches","median"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"TIME:\n";
    for(auto& e:{"optical_flow_slomo","echo_trail","posterize_time","time_displacement"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"SIMULATION:\n";
    for(auto& e:{"particles","shatter","caustics"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"PERSPECTIVE:\n";
    for(auto& e:{"drop_shadow","bevel_edges"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"CHANNEL:\n";
    for(auto& e:{"shift_channels","minimax_dilate","minimax_erode","invert_channel"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"KEYING:\n";
    for(auto& e:{"chroma_key","luma_key"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"CUSTOM AMV:\n";
    for(auto& e:{"deep_glow","motion_blur","datamosh","chromatic","energy_aura",
                  "fire_outline","anamorphic_flare","speed_lines","light_leak",
                  "vignette","vhs","ink_outline","scanlines","color_dodge_aura",
                  "twitch","echo_trail"})
        std::cout<<"  "<<e<<"\n";
    std::cout<<"\nPRESETS:\n";
    for(auto& kv:PRESETS) std::cout<<"  "<<kv.first<<"\n";
    std::cout<<"\nUSAGE:\n"
             <<"  archive_ultimate_effects.exe --effect <name> --input in.mp4 --output out.mp4 [opts]\n"
             <<"  archive_ultimate_effects.exe --preset <name> --input in.mp4 --output out.mp4\n"
             <<"  archive_ultimate_effects.exe --list\n";
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc,char* argv[]){
    std::cout<<"ARCHIVE ULTIMATE EFFECTS ENGINE v"<<EXE_VERSION<<"\n";
    if(argc<2){print_all();return 0;}
    Cfg c=parse(argc,argv);
    if(c.list){print_all();return 0;}
    if(c.input.empty()){std::cerr<<"--input required\n";return 1;}
    if(c.output.empty()){std::cerr<<"--output required\n";return 1;}
    if(!fs::exists(c.input)){std::cerr<<"Input not found: "<<c.input<<"\n";return 1;}
    auto t0=std::chrono::steady_clock::now();
    bool ok=false;
    if(!c.preset.empty()) ok=run_preset(c);
    else if(!c.effect.empty()) ok=run_single(c);
    else{std::cerr<<"--effect or --preset required\n";return 1;}
    double el=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    if(ok) std::cout<<"DONE in "<<el<<"s -> "<<c.output<<"\n";
    else   std::cerr<<"FAILED after "<<el<<"s\n";
    return ok?0:1;
}


