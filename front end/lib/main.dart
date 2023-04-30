import 'package:flutter/material.dart';
import 'package:gradproject/pages/account.dart';
import 'package:gradproject/pages/home.dart';
import 'package:gradproject/pages/login.dart';
import 'package:gradproject/pages/register.dart';
import 'package:gradproject/pages/threed.dart';
import 'package:gradproject/wrapper.dart';
import 'package:provider/provider.dart';
import 'firebase_options.dart';
import 'providers/user_provider.dart';
import 'package:firebase_core/firebase_core.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
  options: DefaultFirebaseOptions.currentPlatform,
);
  runApp(const MyApp());
}




class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    //TODO Uncomment this part of code

    return MultiProvider(
      providers: [
        ChangeNotifierProvider<UserProvider>(
          create: (_) => UserProvider(),
        )
      ],
      child: MaterialApp(
        title: 'Cure',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          primarySwatch: Colors.blue,
          brightness: Brightness.light,
        ),
        initialRoute: '/',
        routes: {
          '/': (context) => Wrapper(),
          '/login': (context) => LogIn(),
          '/register': (context) => SignUp(),
          '/settings': (context) => Account(),
        },
      ),
    );
    
  }
}

