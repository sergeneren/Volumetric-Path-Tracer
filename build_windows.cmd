@echo off

setlocal

git submodule init 
git submodule update

if not exist ./vcpkg/vcpkg.exe (
	call ./vcpkg/bootstrap-vcpkg.bat
) 

if not exist ./vcpkg/vcpkg.exe (
	echo Unable to bootstrap vcpkg! 
	exit 1 
) 

if exist build rmdir /s /q build

cmake --preset Default . 
if errorlevel 1 (
	echo failed to configure
	exit 1
)


cmake --build ./build --preset Default
if errorlevel 1 (
	echo failed to build
	exit 1
)

echo Successfully built VPT! You can find the executable under build/bin directory