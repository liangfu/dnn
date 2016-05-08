#!/usr/bin/env bash

ffmpeg -i original.avi -s 320x240 -vf fps=2. im%04d.png