// import 'package:final_project/models/user.dart';
// import 'package:final_project/providers/user_provider.dart';
// import 'package:font_awesome_flutter/font_awesome_flutter.dart';
// import 'package:provider/provider.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:gradproject/models/user.dart';

//* MY Pages
import 'package:gradproject/pages/home.dart';
import 'package:gradproject/pages/account.dart';
import 'package:gradproject/pages/ThreeDGallery.dart';
import 'package:gradproject/pages/gallery.dart';
import 'package:gradproject/pages/login.dart';
import 'package:gradproject/providers/user_provider.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

class Wrapper extends StatefulWidget {
  Wrapper({Key? key}) : super(key: key);

  @override
  _WrapperState createState() => _WrapperState();
}

class _WrapperState extends State<Wrapper> {
  int currentIndex = 0;
  final List<String> titles = ['Home', 'Search', 'Notifications'];

  final List<Widget> pages = [Home(), Gallery(), ThreeDGallery(), Account()];
  bool signout = false;
  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((timeStamp) async {
      final auth = Provider.of<UserProvider>(context, listen: false);
      final prefs = await SharedPreferences.getInstance();
      String? email = prefs.getString('email');
      String? password = prefs.getString('password');

      if (email != null && password != null) {
        auth.signIn(email, password);
      } else {
        Navigator.pushNamed(context, '/login');
      }
    });

    // Future.delayed(Duration(seconds: 20),(){
    //   Navigator.pushNamed(context, '/login');
    // });
  }

  @override
  Widget build(BuildContext context) {
    User? user = context.watch<UserProvider>().user;
    if (user == null && !signout) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    } else if (user == null && signout) {
      Future.delayed(Duration(milliseconds: 50), () {
        Navigator.pushNamed(context, '/login');
      });
      return SizedBox();
    } else {
      setState(() {
        signout = true;
      });
      return Scaffold(
        appBar: AppBar(
          title: Text('Cure',
              style: GoogleFonts.sacramento(
                  textStyle: const TextStyle(fontSize: 30))),
          actions: [
            IconButton(
                onPressed: () {
                  Navigator.pushNamed(context, '/settings');
                },
                icon: const Icon(Icons.settings)),
          ],
          centerTitle: true,
          backgroundColor: Colors.blue[200],
        ),
        bottomNavigationBar: BottomNavigationBar(
          currentIndex: currentIndex,
          type: BottomNavigationBarType.shifting,
          showSelectedLabels: false,
          showUnselectedLabels: false,
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.camera_enhance, color: Colors.grey),
              activeIcon: Icon(Icons.camera_enhance, color: Colors.black),
              label: "Home",
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.photo, color: Colors.grey),
              activeIcon: Icon(Icons.photo, color: Colors.black),
              label: "Search",
            ),
            BottomNavigationBarItem(
                icon: Icon(Icons.schedule, color: Colors.grey),
                activeIcon: Icon(Icons.schedule, color: Colors.black),
                label: "Notifications"),
            // BottomNavigationBarItem(
            //     icon: FaIcon(FontAwesomeIcons.userAlt, color: Colors.white70),
            //     activeIcon: FaIcon(FontAwesomeIcons.userAlt, color: Colors.white),
            //     label: "Profile"),
          ],
          onTap: (index) {
            setState(() {
              currentIndex = index;
            });
          },
        ),
        body: GestureDetector(
          behavior: HitTestBehavior.translucent,
          onHorizontalDragEnd: (DragEndDetails details) {
            if (details.primaryVelocity! > 0) {
              // User swiped Left
              if (currentIndex != 0) {
                setState(() {
                  currentIndex--;
                });
              }
            } else if (details.primaryVelocity! < 0) {
              // User swiped Right

              if (currentIndex != 2) {
                setState(() {
                  currentIndex++;
                });
              }
            }
          },
          child: pages[currentIndex],
        ),
      );
    }
  }
}
