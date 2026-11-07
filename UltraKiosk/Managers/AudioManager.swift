import AVFoundation
import Combine

// ============================================================================================================
// JP's fork (2026-11-07): voice now runs through Home Assistant's own Assist pipeline, like the Echo does.
//
//   iPad mic (16 kHz mono) --websocket--> HA assist_pipeline/run, start_stage "wake_word"
//        -> wake word "Eden" detected on the SERVER (wyoming-microwakeword, same model as the Echo)
//        -> Whisper STT -> Eden (conversation agent) -> Piper "Alba" voice -> this app plays the reply MP3.
//
// Replaces upstream's Porcupine wake word (paid only since June 2026) + Apple speech recognition (hard-coded
// German) + Apple TTS. No new settings screen: the "Conversation Agent" setting picks the pipeline by NAME or ID
// (e.g. "Riverview (iPad)"); if it matches nothing, the first pipeline that has a wake word is used.
// The mic is paused while a reply plays, so Eden can't wake herself.
// ============================================================================================================

enum APIError: Error, LocalizedError {
    case invalidURL
    case httpError(Int)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid URL"
        case .httpError(let code): return "HTTP Error: \(code)"
        case .invalidResponse: return "Invalid response format"
        }
    }
}

extension APIError: Equatable {}

/// Closure type used to perform a URL data load (kept for upstream's unit tests).
typealias URLDataLoader = (URLRequest) async throws -> (Data, URLResponse)

class AudioManager: NSObject, ObservableObject {

    private let settings = SettingsManager.shared

    @Published var isRecording = false
    @Published var homeAssistantConnected = false
    @Published var isSpeaking = false

    private var preferredNumChannels: Int = 1

    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?
    private var mixerNode: AVAudioMixerNode?
    private var player: AVPlayer?
    private var playbackObserver: NSObjectProtocol?
    private var cuePlayer: AVAudioPlayer?

    private lazy var pipeline = AssistPipelineClient(manager: self)

    private var lastAudioAt = Date()
    private var watchdog: Timer?

    override init() {
        super.init()
        let nc = NotificationCenter.default
        // The mic stops when iOS interrupts or reconfigures audio (camera, Siri, calls, route changes): restart it.
        nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] n in
            guard let self = self,
                  let raw = n.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                  AVAudioSession.InterruptionType(rawValue: raw) == .ended else { return }
            AppLogger.audio.info("Audio interruption ended - restarting mic")
            self.restartMic()
        }
        nc.addObserver(forName: .AVAudioEngineConfigurationChange, object: nil, queue: .main) { [weak self] _ in
            AppLogger.audio.info("Audio engine configuration changed - restarting mic")
            self?.restartMic()
        }
        nc.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: nil, queue: .main) { [weak self] _ in
            self?.setupAudioSession()
            self?.restartMic()
        }
        // Safety net: if no mic audio for 5 s while we should be listening (and nothing is playing), restart it.
        watchdog = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            guard let self = self, self.isRecording, !self.isSpeaking,
                  Date().timeIntervalSince(self.lastAudioAt) > 5 else { return }
            AppLogger.audio.warning("No mic audio for 5 s - restarting mic")
            self.restartMic()
        }
    }

    private var lastRestart = Date.distantPast

    private func restartMic() {
        // Debounced: restarting the engine can itself post a configuration-change notification.
        guard settings.enableVoiceActivation, player == nil,
              Date().timeIntervalSince(lastRestart) > 3 else { return }
        lastRestart = Date()
        stopRecording()
        setupAudioSession()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { self.startRecording() }
    }

    // MARK: - Short cues ("listening" / "done") - played WITHOUT restarting the mic

    func playCue(_ name: String) {
        guard let url = Bundle.main.url(forResource: name, withExtension: "wav") else { return }
        do {
            cuePlayer = try AVAudioPlayer(contentsOf: url)
            cuePlayer?.play()
        } catch {
            AppLogger.audio.error("Cue \(name) failed: \(String(describing: error))")
        }
    }

    // Upstream API, kept for compatibility (restarts recording after the sound, like before).
    func playFeedbackSound(assetName: String = "bing", assetExtension: String = "caf") {
        if let url = Bundle.main.url(forResource: assetName, withExtension: assetExtension) {
            playURL(url)
        }
    }

    // MARK: - Audio session + engine

    func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playAndRecord, mode: .default,
                                         options: [.defaultToSpeaker, .allowBluetoothA2DP])
            try audioSession.setPreferredSampleRate(Double(settings.voiceSampleRate))
            if audioSession.isInputAvailable {
                try? audioSession.setPreferredInputNumberOfChannels(preferredNumChannels)
            }
            try audioSession.setActive(true)
        } catch {
            AppLogger.audio.error("Audio session setup failed: \(String(describing: error))")
        }
    }

    func startRecording() {
        audioEngine = AVAudioEngine()
        inputNode = audioEngine?.inputNode
        guard let audioEngine = audioEngine, let inputNode = inputNode else { return }

        // 16 kHz mono float from the tap; converted to 16-bit PCM for Home Assistant.
        let desiredFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                          sampleRate: 16000,
                                          channels: AVAudioChannelCount(preferredNumChannels),
                                          interleaved: false)

        let mixer = AVAudioMixerNode()
        mixerNode = mixer
        audioEngine.attach(mixer)
        let hardwareFormat = inputNode.outputFormat(forBus: 0)
        audioEngine.connect(inputNode, to: mixer, format: hardwareFormat)
        audioEngine.connect(mixer, to: audioEngine.mainMixerNode, format: desiredFormat)
        audioEngine.mainMixerNode.outputVolume = 0          // never play the mic through the speaker

        mixer.installTap(onBus: 0, bufferSize: 1024, format: desiredFormat) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer)
        }

        do {
            try audioEngine.start()
            DispatchQueue.main.async { self.isRecording = true }
            pipeline.start()                                  // no-op if already connected
        } catch {
            AppLogger.audio.error("Failed to start audio engine: \(String(describing: error))")
        }
    }

    func stopRecording() {
        audioEngine?.stop()
        mixerNode?.removeTap(onBus: 0)
        mixerNode = nil
        inputNode = nil
        audioEngine = nil
        DispatchQueue.main.async { self.isRecording = false }
    }

    func convertFloatToInt16(floatSamples: [Float]) -> [Int16] {
        var out = [Int16](repeating: 0, count: floatSamples.count)
        for i in 0..<floatSamples.count {
            let v = max(-1.0, min(1.0, floatSamples[i]))
            out[i] = Int16(v * Float(Int16.max))
        }
        return out
    }

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channel = buffer.floatChannelData?[0] else { return }
        lastAudioAt = Date()
        let n = Int(buffer.frameLength)
        if n == 0 { return }
        let samples = convertFloatToInt16(floatSamples: Array(UnsafeBufferPointer(start: channel, count: n)))
        let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }   // little-endian on iOS
        pipeline.sendAudio(data)
    }

    // MARK: - Reply playback (Eden's voice from Home Assistant)

    func playMP3(from url: URL) {
        playURL(url)
    }

    private func playURL(_ url: URL) {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setActive(true)
            try session.overrideOutputAudioPort(.speaker)
        } catch {
            AppLogger.audio.error("Audio session activation/override failed: \(String(describing: error))")
        }
        DispatchQueue.main.async { self.isSpeaking = true }
        let item = AVPlayerItem(url: url)
        player = AVPlayer(playerItem: item)
        playbackObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime, object: item, queue: .main
        ) { [weak self] _ in
            self?.handlePlaybackCompleted()
        }
        // If the reply can't be played (bad URL, network), don't leave the mic paused forever.
        NotificationCenter.default.addObserver(forName: .AVPlayerItemFailedToPlayToEndTime, object: item, queue: .main) { [weak self] _ in
            AppLogger.audio.error("Reply playback failed")
            self?.handlePlaybackCompleted()
        }
        player?.play()
    }

    private func handlePlaybackCompleted() {
        if let observer = playbackObserver {
            NotificationCenter.default.removeObserver(observer)
            playbackObserver = nil
        }
        player = nil
        DispatchQueue.main.async { self.isSpeaking = false }
        do {
            try AVAudioSession.sharedInstance().overrideOutputAudioPort(.none)
            setupAudioSession()
        } catch {
            AppLogger.audio.error("Failed to restore audio session after playback: \(String(describing: error))")
        }
        if isRecording {                                      // the speaker override can break the input: restart it
            stopRecording()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { self.startRecording() }
        }
        pipeline.replyFinished()                              // listen for the wake word again
    }

    // MARK: - Upstream text API (unused by the voice path now; kept so upstream's tests still compile)

    func sendHomeAssistantConversation(
        text: String,
        language: String,
        urlLoader: URLDataLoader = { try await URLSession.shared.data(for: $0) }
    ) async throws -> String {
        guard let url = URL(string: "\(settings.homeAssistantBaseURL)/api/conversation/process") else {
            throw APIError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(settings.accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        let body: [String: Any] = ["text": text, "language": language,
                                   "agent_id": settings.homeAssistantConversationAgent,
                                   "conversation_id": settings.homeAssistantConversationId]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await urlLoader(request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw APIError.httpError((response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        guard let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let r = json["response"] as? [String: Any], let speech = r["speech"] as? [String: Any],
              let plain = speech["plain"] as? [String: Any], let text = plain["speech"] as? String else {
            throw APIError.invalidResponse
        }
        return text
    }

    fileprivate func setConnected(_ value: Bool) {
        DispatchQueue.main.async { self.homeAssistantConnected = value }
    }
}

// ============================================================================================================
// Home Assistant websocket client for one continuous voice satellite.
// One pipeline run at a time: wake_word -> stt -> intent -> tts. When a run ends we start the next one; while
// Eden's reply is playing we send no audio (paused) and start the next run only after playback finishes.
// ============================================================================================================
final class AssistPipelineClient: NSObject {
    private weak var manager: AudioManager?
    private let settings = SettingsManager.shared
    private let queue = DispatchQueue(label: "ultrakiosk.assist")

    private var socket: URLSessionWebSocketTask?
    private var session: URLSession?
    private var started = false
    private var authed = false
    private var nextId = 1
    private var listId = -1
    private var runId = -1
    private var pipelineId: String?
    private var handlerId: UInt8?
    private var runActive = false
    private var waitingForReply = false          // TTS url received, reply playing: no audio, no new run

    init(manager: AudioManager) {
        self.manager = manager
        super.init()
    }

    func start() {
        queue.async {
            if self.started { return }
            self.started = true
            self.connect()
        }
    }

    func sendAudio(_ pcm: Data) {
        queue.async {
            guard self.runActive, !self.waitingForReply, let h = self.handlerId, let s = self.socket else { return }
            var msg = Data([h])
            msg.append(pcm)
            s.send(.data(msg)) { _ in }
        }
    }

    func replyFinished() {
        queue.async {
            self.waitingForReply = false
            if !self.runActive { self.queue.asyncAfter(deadline: .now() + 0.4) { self.startRun() } }
        }
    }

    // MARK: connection

    private func baseURL() -> String { settings.homeAssistantBaseURL }

    private func connect() {
        let ws = baseURL().replacingOccurrences(of: "https://", with: "wss://")
                          .replacingOccurrences(of: "http://", with: "ws://") + "/api/websocket"
        guard let url = URL(string: ws) else {
            AppLogger.homeAssistant.error("Bad Home Assistant URL for websocket: \(ws)")
            return
        }
        authed = false; runActive = false; handlerId = nil; pipelineId = nil
        session = URLSession(configuration: .default)
        socket = session?.webSocketTask(with: url)
        socket?.maximumMessageSize = 4 * 1024 * 1024
        socket?.resume()
        AppLogger.homeAssistant.info("Assist: connecting to \(ws)")
        receive()
    }

    private func reconnect(after seconds: Double) {
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        manager?.setConnected(false)
        queue.asyncAfter(deadline: .now() + seconds) { self.connect() }
    }

    private func send(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let text = String(data: data, encoding: .utf8) else { return }
        socket?.send(.string(text)) { error in
            if let error = error { AppLogger.homeAssistant.error("Assist send failed: \(error.localizedDescription)") }
        }
    }

    private func receive() {
        socket?.receive { [weak self] result in
            guard let self = self else { return }
            self.queue.async {
                switch result {
                case .failure(let error):
                    AppLogger.homeAssistant.error("Assist websocket closed: \(error.localizedDescription)")
                    self.reconnect(after: 5)
                case .success(let message):
                    if case .string(let text) = message,
                       let data = text.data(using: .utf8),
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        self.handle(json)
                    }
                    self.receive()
                }
            }
        }
    }

    // MARK: protocol

    private func handle(_ m: [String: Any]) {
        let type = m["type"] as? String ?? ""
        switch type {
        case "auth_required":
            send(["type": "auth", "access_token": settings.accessToken])
        case "auth_ok":
            authed = true
            manager?.setConnected(true)
            listId = nextId; nextId += 1
            send(["id": listId, "type": "assist_pipeline/pipeline/list"])
        case "auth_invalid":
            AppLogger.homeAssistant.error("Assist: access token rejected - check the token in settings")
            reconnect(after: 60)
        case "result":
            let id = m["id"] as? Int ?? -1
            let ok = m["success"] as? Bool ?? false
            if id == listId {
                choosePipeline(m["result"] as? [String: Any])
                startRun()
            } else if id == runId && !ok {
                let err = (m["error"] as? [String: Any])?["message"] as? String ?? "?"
                AppLogger.homeAssistant.error("Assist: run refused: \(err)")
                runActive = false
                queue.asyncAfter(deadline: .now() + 10) { self.startRun() }
            }
        case "event":
            guard (m["id"] as? Int) == runId, let ev = m["event"] as? [String: Any] else { return }
            handleEvent(ev["type"] as? String ?? "", ev["data"] as? [String: Any] ?? [:])
        default:
            break
        }
    }

    private func choosePipeline(_ result: [String: Any]?) {
        let list = result?["pipelines"] as? [[String: Any]] ?? []
        let wanted = settings.homeAssistantConversationAgent.trimmingCharacters(in: .whitespaces)
        if let p = list.first(where: { ($0["name"] as? String) == wanted || ($0["id"] as? String) == wanted }) {
            pipelineId = p["id"] as? String
        } else if let p = list.first(where: { !(($0["wake_word_entity"] as? String) ?? "").isEmpty }) {
            pipelineId = p["id"] as? String
        } else {
            pipelineId = nil
        }
        AppLogger.homeAssistant.info("Assist: using pipeline \(self.pipelineId ?? "<none - add a wake word to a pipeline>")")
    }

    private func startRun() {
        guard authed, !runActive, !waitingForReply, let pid = pipelineId else { return }
        runId = nextId; nextId += 1
        handlerId = nil
        runActive = true
        send(["id": runId, "type": "assist_pipeline/run",
              "start_stage": "wake_word", "end_stage": "tts", "pipeline": pid,
              "input": ["sample_rate": 16000, "timeout": 3600],   // seconds of SILENCE before giving up (HA default 3)
              "timeout": 3600])
    }

    private func handleEvent(_ type: String, _ d: [String: Any]) {
        switch type {
        case "run-start":
            if let rd = d["runner_data"] as? [String: Any], let h = rd["stt_binary_handler_id"] as? Int {
                handlerId = UInt8(truncatingIfNeeded: h)
            }
        case "wake_word-end":
            if let out = d["wake_word_output"] as? [String: Any], !out.isEmpty {
                AppLogger.speech.info("Assist: wake word \(String(describing: out["wake_word_id"]))")
                DispatchQueue.main.async { self.manager?.playCue("Speech_On") }
            }
        case "stt-end":
            let text = (d["stt_output"] as? [String: Any])?["text"] as? String ?? ""
            AppLogger.speech.info("Assist: heard '\(text)'")
            DispatchQueue.main.async { self.manager?.playCue("Speech_Off") }
        case "tts-end":
            if let out = d["tts_output"] as? [String: Any], let u = out["url"] as? String {
                let full = u.hasPrefix("http") ? u : baseURL() + u
                if let url = URL(string: full) {
                    waitingForReply = true
                    DispatchQueue.main.async { self.manager?.playMP3(from: url) }
                }
            }
        case "error":
            AppLogger.homeAssistant.warning("Assist: \(String(describing: d["code"])) \(String(describing: d["message"]))")
        case "run-end":
            runActive = false
            handlerId = nil
            if !waitingForReply { queue.asyncAfter(deadline: .now() + 0.3) { self.startRun() } }
        default:
            break
        }
    }
}
