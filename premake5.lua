
workspace "Open-SDF"
    location  "Build"
    targetdir ("%{wks.location}/%{prj.name}/%{cfg.longname}")
    objdir ("%{wks.location}/Temp/%{prj.name}/%{cfg.longname}")
    language            "C++"
    cppdialect          "C++20"
    architecture        "x86_64"
    vectorextensions    "AVX"
    staticruntime       "On" 
    startproject        "Measure"
    
    flags {
        "MultiProcessorCompile", 
        "NoBufferSecurityCheck",
        "NoIncrementalLink",
        "NoManifest",
        "NoMinimalRebuild",
        "NoPCH",
    }
    
    externalincludedirs {
        "External/glm",
    }
    
    configurations { 
        "Dev-Debug", 
        "Dev-Run",
    }

    defines {
        "GLM_FORCE_DEPTH_ZERO_TO_ONE=1",
        "GLM_FORCE_INLINE=1",
    }
        
    filter { "configurations:Dev-Debug" }
        symbols  "On"
        optimize "Debug"
        defines{
            "_DEBUG",
        }
    
    filter { "configurations:Dev-Run" }
        symbols  "On"
        optimize "Speed"
        defines{
            "NDEBUG",
        }
    
----------------------------------------------------------------------------------------------------------
project "Measure"
    
    kind "ConsoleApp"
    targetname "measure"


    files {
        "Source/Measure/**.cpp",
        "Source/Measure/**.c",
        "Source/Measure/**.hpp",
        "Source/Measure/**.h",
        "Source/Measure/**.inl",
    }

    includedirs {
        "Source/Measure",
        "Source/Functions",
    }
    
    libdirs { 
    }
    
    