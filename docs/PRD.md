# Product Requirements Document (PRD)
## SentinelAI — Driver Drowsiness Detection System

**Version**: 1.0  
**Status**: Complete

---

## Problem Statement

Driver fatigue is a leading cause of road accidents worldwide. Existing solutions are expensive hardware units. This project demonstrates a software-only approach using a standard webcam and computer vision to detect early signs of drowsiness in real time.

---

## Goals

1. Detect drowsiness from webcam feed with low latency
2. Alert driver visually through a clear dashboard UI
3. Track session-level fatigue events for review
4. Keep the implementation understandable for a college-level audience

---

## User Stories

- As a driver, I want to be alerted when my eyes are closing so I can take a break.
- As a driver, I want to see my eye and mouth metrics in real time.
- As a user, I want to start and stop monitoring with one click.
- As a user, I want a session summary after driving.

---

## Features (v1.0)

| Feature | Priority | Status |
|---------|----------|--------|
| Face detection | P0 | Done |
| Eye closure detection (EAR) | P0 | Done |
| Yawn detection (MAR) | P0 | Done |
| DROWSY / YAWNING / AWAKE status | P0 | Done |
| Live metric dashboard | P0 | Done |
| MJPEG video stream | P0 | Done |
| Alert banner | P1 | Done |
| Session event counters | P1 | Done |
| Risk scoring | P1 | Done |
| Session report tab | P1 | Done |
| Annotated video overlay | P1 | Done |

---

## Out of Scope (v1.0)

- Audio alerts
- Multi-camera support
- Multiple face tracking
- Data persistence / export
- Authentication
- Mobile app
