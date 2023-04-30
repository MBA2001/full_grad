import 'package:babylonjs_viewer/babylonjs_viewer.dart';
import 'package:flutter/material.dart';
import 'package:gradproject/providers/user_provider.dart';
import 'package:provider/provider.dart';
import 'package:webview_flutter/webview_flutter.dart';

class ThreeDViewer extends StatefulWidget {
  final int xx;
  ThreeDViewer({Key? key, required this.xx}) : super(key: key);
  @override
  State<ThreeDViewer> createState() => _ThreeDViewerState();
}

class _ThreeDViewerState extends State<ThreeDViewer> {
  @override
  Widget build(BuildContext context) {
    final userpro = Provider.of<UserProvider>(context);
    return Scaffold(
        body: BabylonJSViewer(
            src: 'assets/girl_mohamedd.glb'));
  }
}
