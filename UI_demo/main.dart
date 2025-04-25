import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ObstacleDemoUI(),
    );
  }
}

class ObstacleDemoUI extends StatefulWidget {
  @override
  _ObstacleDemoUIState createState() => _ObstacleDemoUIState();
}

class _ObstacleDemoUIState extends State<ObstacleDemoUI> {
  bool _detecting = false;
  String _statusText = "Tap to start detection";
  String _fakeResult = "Detected: pedestrian, lamppost";
  FlutterTts _tts = FlutterTts();

  void _toggleDetection() {
    setState(() {
      _detecting = !_detecting;
      _statusText = _detecting ? "Detecting..." : "Detection stopped";
    });
  }

  void _speakResult() async {
    await _tts.speak(_fakeResult);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Obstacle Detection Demo")),
      body: Stack(
        children: [
          Positioned.fill(
            child: Image.asset(
              'assets/fake_camera.jpg',
              fit: BoxFit.cover,
            ),
          ),
          Positioned(
            bottom: 140,
            left: 20,
            right: 20,
            child: Container(
              padding: EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.5),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _fakeResult,
                style: TextStyle(color: Colors.white, fontSize: 20),
                textAlign: TextAlign.center,
              ),
            ),
          ),
          Positioned(
            bottom: 80,
            left: 20,
            right: 20,
            child: ElevatedButton.icon(
              onPressed: _speakResult,
              icon: Icon(Icons.volume_up),
              label: Text("Speak Detection Result"),
            ),
          ),
          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: ElevatedButton.icon(
              onPressed: _toggleDetection,
              icon: Icon(_detecting ? Icons.stop : Icons.play_arrow),
              label: Text(_detecting ? "Stop Detection" : "Start Detection"),
              style: ElevatedButton.styleFrom(
                backgroundColor: _detecting ? Colors.red : Colors.green,
              ),
            ),
          ),
        ],
      ),
    );
  }
}