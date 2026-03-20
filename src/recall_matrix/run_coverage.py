import argparse
import sys
import unittest
from pathlib import Path

import coverage


def run_coverage(html=False, xml=False):
    """Run unit tests with coverage reporting."""

    # Find the project root by looking for pyproject.toml
    current = Path(__file__).absolute()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            project_root = current
            break
        current = current.parent
    else:
        raise RuntimeError("Could not find project root (pyproject.toml not found)")

    # Initialize coverage
    cov = coverage.Coverage(config_file=str(project_root / ".coveragerc"))
    cov.start()

    try:
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.discover(
            start_dir=str(project_root / "src/recall_matrix"),
            pattern="unit_tests.py",
            top_level_dir=str(project_root),
        )

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # Stop coverage and generate report
        cov.stop()
        cov.save()

        # Print console report
        print("\n" + "=" * 70)
        print("COVERAGE REPORT")
        print("=" * 70 + "\n")
        cov.report()

        # Generate HTML report if requested
        if html:
            print("\n" + "=" * 70)
            print("Generating HTML coverage report...")
            html_report_dir = project_root / "htmlcov"
            cov.html_report(directory=str(html_report_dir))
            html_index = html_report_dir / "index.html"
            print(f"HTML report generated at: {html_index}")
            print()

        # Generate XML report if requested
        if xml:
            print("\n" + "=" * 70)
            print("Generating XML coverage report...")
            xml_file = project_root / "coverage.xml"
            cov.xml_report(outfile=str(xml_file))
            print(f"XML report generated at: {xml_file}")
            print()

        # Return exit code based on test results
        return 0 if result.wasSuccessful() else 1

    except Exception as e:
        print(f"Error running tests: {e}", file=sys.stderr)
        cov.stop()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run unit tests with coverage reporting"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report",
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate XML coverage report for CI/CD",
    )

    args = parser.parse_args()

    sys.exit(run_coverage(html=args.html, xml=args.xml))
