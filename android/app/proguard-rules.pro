# minifyEnabled is false for the measurement build, so these rules are inert
# today. They exist so that turning minification on later cannot silently strip
# the ORT JNI bridge and turn a working scorer into a runtime failure.
-keep class ai.onnxruntime.** { *; }
-keep class org.deploybench.app.** { *; }
