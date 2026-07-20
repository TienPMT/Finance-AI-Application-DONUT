enum AppMode { demo, full }

class AppConfig {
  static AppMode mode = AppMode.full;

  static const String _apiBase = String.fromEnvironment(
    "API_BASE",
    defaultValue: "http://localhost:8081",
  );
  static const String _mlBase = String.fromEnvironment(
    "ML_BASE",
    defaultValue: "http://localhost:8000",
  );

  static String get apiBaseUrl => _apiBase;
  static String get mlApiUrl => _mlBase;
  static const String apiVersion = "/api";

  static const String googleServerClientId =
      "69082235004-93c8c9fos2ts92hicce0uth5gedo7hk0.apps.googleusercontent.com";

  static bool get isDemo => mode == AppMode.demo;
  static bool get isFull => mode == AppMode.full;
}
