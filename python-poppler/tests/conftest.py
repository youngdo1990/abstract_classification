# poppler-python: python binding to the poppler-cpp pdf lib
# Copyright (C) 2020, Charles Brunet <charles@cbrunet.net>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import pytest

from pathlib import Path

from poppler import load_from_file


@pytest.fixture(scope="session")
def data_path():
    return Path(__file__).parent / "data"


@pytest.fixture()
def pdf_document(data_path):
    return load_from_file(data_path / "document.pdf", "owner", "user")


@pytest.fixture
def pdf_page(pdf_document):
    return pdf_document.create_page(0)


@pytest.fixture(scope="session")
def sample_document(data_path):
    return load_from_file(data_path / "sample.pdf")


@pytest.fixture()
def document_with_error(data_path):
    return load_from_file(data_path / "error_log.pdf")
