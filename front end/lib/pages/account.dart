import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:gradproject/providers/user_provider.dart';
import 'package:provider/provider.dart';

class Account extends StatefulWidget {
  Account({Key? key}) : super(key: key);

  @override
  State<Account> createState() => _AccountState();
}

class _AccountState extends State<Account> {
  @override
  Widget build(BuildContext context) {
    final userProvider = Provider.of<UserProvider>(context);
    return Scaffold(
      appBar: AppBar(
        title: Text('Cure',
            style: GoogleFonts.sacramento(textStyle: TextStyle(fontSize: 30))),
        centerTitle: true,
        backgroundColor: Colors.blue[200],
      ),
      body: Center(
        child: Column(
          children: [
            SizedBox(
                width: double.infinity,
                child: Padding(
                  padding: const EdgeInsets.only(left: 2),
                  child: Text(
                    'Account Actions',
                    textAlign: TextAlign.left,
                    style: GoogleFonts.poppins(textStyle: TextStyle(fontSize: 20,fontWeight: FontWeight.bold)),
                  ),
                )),
            Expanded(
              child: ListView(
                children: [
                  Card(
                    elevation: 5,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: ListTile(
                      leading: const Icon(Icons.image),
                      title: const Text('Change Image'),
                      trailing: const Icon(Icons.arrow_forward_ios),
                      onTap: () {},
                    ),
                  ),
                  Card(
                    elevation: 5,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: ListTile(
                      leading: const Icon(Icons.person),
                      title: const Text('Change Username'),
                      trailing: const Icon(Icons.arrow_forward_ios),
                      onTap: () {},
                    ),
                  ),
                  Card(
                    elevation: 5,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: ListTile(
                      leading: const Icon(Icons.email),
                      title: const Text('Change Email'),
                      trailing: const Icon(Icons.arrow_forward_ios),
                      onTap: () {},
                    ),
                  ),
                  Card(
                    elevation: 5,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: ListTile(
                      leading: const Icon(Icons.password),
                      title: const Text('Change Password'),
                      trailing: const Icon(Icons.arrow_forward_ios),
                      onTap: () {},
                    ),
                  ),
                  SizedBox(
                width: double.infinity,
                child: Padding(
                  padding: const EdgeInsets.only(left: 2),
                  child: Text(
                    'Account Actions',
                    textAlign: TextAlign.left,
                    style: GoogleFonts.poppins(textStyle: TextStyle(fontSize: 20,fontWeight: FontWeight.bold)),
                  ),
                )),
                Card(
                  elevation: 5,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: ListTile(
                      leading: const Icon(Icons.clear_all),
                      title: const Text('Clear Cache'),
                      trailing: const Icon(Icons.arrow_forward_ios),
                      onTap: () {},
                      style: ListTileStyle.list,
                    ),
                  ),
                  Card(
                    elevation: 5,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: ListTile(
                      leading: const Icon(Icons.delete_forever),
                      title: const Text('Delete Account'),
                      trailing: const Icon(Icons.arrow_forward_ios),
                      onTap: () {},
                    ),
                  ),
                  Card(
                    elevation: 5,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: ListTile(
                      leading: const Icon(Icons.logout),
                      title: const Text('Logout'),
                      trailing: const Icon(Icons.arrow_forward_ios),
                      onTap: () {
                        userProvider.signOut();
                        Navigator.pop(context);
                      },
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
