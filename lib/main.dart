import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma_litertlm/flutter_gemma_litertlm.dart';
import 'package:flutter/services.dart';
void main() {
  WidgetsFlutterBinding.ensureInitialized();
  
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple, 
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple, 
          brightness: Brightness.dark, 
        ),
        useMaterial3: true,
      ),
      themeMode: ThemeMode.system,
      home: const NotePad(),
    );
  }
}

class NotePad extends StatefulWidget {
  const NotePad({super.key});

  @override
  State<NotePad> createState() => _NotePadState();
}

class _NotePadState extends State<NotePad> {
  final TextEditingController _myController = TextEditingController();
  
  // Track engine initialization and download progress
  bool _isModelReady = false;
  bool _isProcessing = false;
  String _statusText = "Waking up inference engine...";
  double _downloadProgress = 0.0;

  @override
  void initState() {
    super.initState();
    _initGemmaAsync();
  }

  Future<void> _initGemmaAsync() async {
    try {
      // Initialize the LiteRT backend
      await FlutterGemma.initialize(
        inferenceEngines: [LiteRtLmEngine()]
      );

      setState(() {
        _statusText = "Downloading Gemma 4...";
      });

      await FlutterGemma.installModel(
        modelType: ModelType.gemmaIt, 
        fileType: ModelFileType.litertlm,
      )
      .fromNetwork('https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm')
      //.fromNetwork('https://huggingface.co/litert-community/Phi-4-mini-instruct/resolve/main/Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm?download=true', foreground: true)
      //.fromNetwork('https://huggingface.co/litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm')
      //.fromNetwork('https://huggingface.co/litert-community/Qwen3-4B-Instruct-2507/resolve/main/qwen3_4b_instruct_2507_mixed_int4.litertlm')
      .withProgress((progress) {
        setState(() {
          _downloadProgress = progress / 100.0;
          _statusText = "Downloading Gemma 4: ${progress.toStringAsFixed(0)}%";
        });
      })
      .install();

      setState(() {
        _isModelReady = true;
      });
    } catch (e) {
      setState(() {
        _statusText = "Error: $e";
      });
    }
  }

  @override
  void dispose() {
    _myController.dispose();
    super.dispose();
  }

  void _askGemma() async {
    if (!_isModelReady || _isProcessing) return;

    String textToSummarize = _myController.text;
    if (textToSummarize.trim().isEmpty) return;

    setState(() {
      _isProcessing = true;
      _statusText = "Summarizing context...";
      _myController.text = "";
    });

    String query = "Summarize the following text into concise bullet points. Use plain text only, no markdown, no asterisks, no special formatting. Use a dash (-) for each bullet. Keep each point short and direct. Respond in the same language as the text.\n\n$textToSummarize";    
    
    try {
      final model = await FlutterGemma.getActiveModel(maxTokens: 8192,
  
      
      );
      final chat = await model.createChat(temperature: 0.7);
      await chat.addQueryChunk(Message(text: query, isUser: true));
      await for (final response in chat.generateChatResponseAsync()) {
        if (response is TextResponse) {
          setState(() {
            _myController.text += response.token;
          });
        }
      }
      
      await chat.close();
      await model.close();
      
      
    } catch (e) {
      setState(() {
        _myController.text = "Inference Exception: $e";
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });

    }
  }



  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: (!_isModelReady || _isProcessing) 
          ? AppBar(
              title: Text(_statusText, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
              bottom: PreferredSize(
                preferredSize: const Size.fromHeight(4),
                child: LinearProgressIndicator(value: _downloadProgress > 0 ? _downloadProgress : null),
              ),
            )
          : null,
      body: SafeArea(
        child: Container(
          width: double.infinity,
          height: double.infinity,
          padding: const EdgeInsets.all(8),
          child: TextField(
            controller: _myController,
            enabled: _isModelReady,
            decoration: InputDecoration(
              hintText: _isModelReady ? "Type or paste your text here to summarize..." : "Please wait, preparing local model...",
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12.0)
              ),
              
              
            ),
            maxLength: null,
            maxLines: null,
            expands: true,
            textAlignVertical: TextAlignVertical.top,
          ),
        ),
      ),
      floatingActionButton: Row(
        mainAxisSize: MainAxisSize.min,
        
        children: [
          FloatingActionButton(
                
                onPressed: _isProcessing ? null : () async {
                    
                    final data = await Clipboard.getData(Clipboard.kTextPlain);
                    if (data?.text != null) {
                    _myController.text = data!.text!;
                    }
                },
                child: const Icon(Icons.paste)
          ),
          const SizedBox(width: 16),
          FloatingActionButton(
            onPressed: _isModelReady ? _askGemma : null,
            backgroundColor: _isModelReady ? null : Colors.grey.withValues(alpha: 0.3),
            child: (!_isModelReady || _isProcessing)
                ? const Padding(
                    padding: EdgeInsets.all(12.0),
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.psychology),
                
      ),
        ]
      ),
      
      floatingActionButtonLocation: FloatingActionButtonLocation.startFloat,
    );
  }
}