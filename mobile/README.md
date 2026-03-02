# TradeMaster Mobile

React Native mobile application for TradeMaster trading platform.

## Setup

```bash
npx react-native init TradeMasterMobile --template react-native-template-typescript
cd TradeMasterMobile
npm install
```

## Architecture

- Shares business logic with web frontend via shared types
- Push notifications for trade alerts and signals
- Biometric authentication (FaceID/TouchID)
- Real-time WebSocket price streaming

## Features

- [ ] Dashboard with portfolio overview
- [ ] Real-time price charts
- [ ] Trade execution (paper + live)
- [ ] Push notifications for alerts
- [ ] Biometric login
- [ ] Offline data caching

## Development

```bash
# iOS
npx react-native run-ios

# Android
npx react-native run-android
```
