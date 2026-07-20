import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:google_mlkit_document_scanner/google_mlkit_document_scanner.dart';

import 'package:finance_ai_app/core/theme/app_theme.dart';
import 'package:finance_ai_app/screens/dashboard/dashboard_page.dart';
import 'package:finance_ai_app/screens/ledger/ledger_page.dart';
import 'package:finance_ai_app/screens/ai/ai_tab_page.dart';
import 'package:finance_ai_app/screens/profile/profile_page.dart';
import 'package:finance_ai_app/screens/receipt_scanner/review_form_page.dart';

class MainScaffold extends StatefulWidget {
  const MainScaffold({super.key});

  @override
  State<MainScaffold> createState() => _MainScaffoldState();
}

class _MainScaffoldState extends State<MainScaffold> {
  int _currentIndex = 0;

  final _ledgerKey = GlobalKey<LedgerPageState>();
  final _dashboardKey = GlobalKey<DashboardPageState>();

  late final List<Widget> _pages = [
    DashboardPage(key: _dashboardKey),
    LedgerPage(key: _ledgerKey),
    const AITabPage(),
    const ProfilePage(),
  ];

  void _showScannerOptions(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppTheme.navyLight,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) {
        return Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: AppTheme.textGray.withValues(alpha: 0.3),
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              const SizedBox(height: 24),
              const Text(
                'Quét hóa đơn bằng AI',
                style: TextStyle(
                  color: AppTheme.textWhite,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 24),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _buildScanOptionBtn(
                    context,
                    Icons.document_scanner_rounded,
                    'Quét AI',
                    () => _startAutoScanner(context),
                  ),
                  _buildScanOptionBtn(
                    context,
                    Icons.photo_library_rounded,
                    'Từ thư viện',
                    () => _pickImageAndScan(context, ImageSource.gallery),
                  ),
                ],
              ),
              const SizedBox(height: 20),
            ],
          ),
        );
      },
    );
  }

  Future<void> _startAutoScanner(BuildContext context) async {
    final DocumentScanner documentScanner = DocumentScanner(
      options: DocumentScannerOptions(
        documentFormats: {DocumentFormat.jpeg},
        mode: ScannerMode.base,
        pageLimit: 1,
        isGalleryImport: true,
      ),
    );
    try {
      final result = await documentScanner.scanDocument();
      final images = result.images;
      if (images != null && images.isNotEmpty && context.mounted) {
        Navigator.pop(context);
        final saved = await Navigator.push<bool>(
          context,
          MaterialPageRoute(
            builder: (_) => ReviewFormPage(imagePath: images.first),
          ),
        );
        if (saved == true) {
          _ledgerKey.currentState?.reload();
          setState(() => _currentIndex = 1);
        }
      }
    } catch (e) {
      debugPrint('Lỗi quét tài liệu ML Kit: $e');
    } finally {
      documentScanner.close();
    }
  }

  Future<void> _pickImageAndScan(
    BuildContext context,
    ImageSource source,
  ) async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: source);

    if (image != null && context.mounted) {
      Navigator.pop(context);
      final saved = await Navigator.push<bool>(
        context,
        MaterialPageRoute(
          builder: (_) => ReviewFormPage(imagePath: image.path),
        ),
      );
      if (saved == true) {
        _ledgerKey.currentState?.reload();
        setState(() => _currentIndex = 1);
      }
    }
  }

  Widget _buildScanOptionBtn(
    BuildContext context,
    IconData icon,
    String label,
    VoidCallback onTap,
  ) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        children: [
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              color: AppTheme.primary.withValues(alpha: 0.10),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: AppTheme.primary, size: 32),
          ),
          const SizedBox(height: 12),
          Text(
            label,
            style: const TextStyle(color: AppTheme.textWhite, fontSize: 14),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBody: true,
      body: IndexedStack(index: _currentIndex, children: _pages),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      floatingActionButton: FloatingActionButton(
        onPressed: () => _showScannerOptions(context),
        backgroundColor: AppTheme.mintAccent,
        elevation: 4,
        child: const Icon(
          Icons.document_scanner_rounded,
          color: AppTheme.navyDark,
          size: 28,
        ),
      ),
      bottomNavigationBar: Container(
        height: 90,
        margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(30),
          child: Container(
            decoration: BoxDecoration(
              color: AppTheme.mintAccent,
              borderRadius: BorderRadius.circular(30),
              boxShadow: AppTheme.softShadow,
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildNavItem(Icons.home_rounded, 'Trang chủ', 0),
                _buildNavItem(Icons.bar_chart_rounded, 'Thống kê', 1),
                const SizedBox(width: 48),
                _buildNavItem(Icons.auto_awesome_rounded, 'AI', 2),
                _buildNavItem(Icons.person_rounded, 'Cá nhân', 3),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNavItem(IconData icon, String label, int index) {
    final isSelected = _currentIndex == index;
    final color = isSelected ? Colors.white : Colors.white54;
    return Expanded(
      child: InkWell(
        onTap: () {
          setState(() => _currentIndex = index);
          if (index == 0) _dashboardKey.currentState?.reload();
          if (index == 1) _ledgerKey.currentState?.reload();
        },
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: color, size: 26),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                color: color,
                fontSize: 10,
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
