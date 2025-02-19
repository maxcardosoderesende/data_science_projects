#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import csv
import datetime
import os
import time

import pandas as pd
import pyexcel_xls
import requests
from tqdm import tqdm


def archive_old_files():
    base_directory = r"C:\Users\Max\Desktop\Max_Data_science\ANP_Gasolina"
    archive_directory = os.path.join(base_directory, "archive")

    # Create the archive folder if it doesn't exist
    if not os.path.exists(archive_directory):
        os.makedirs(archive_directory)

    # Get list of files in the base directory
    file_list = os.listdir(base_directory)
    today = datetime.datetime.now().strftime("%d.%m.%Y")

    for file_name in file_list:
        if not file_name.endswith(".xls"):
            continue

        # Extract the date from the file name
        data_arquivo = file_name.split(".xls")[-2][-10:]

        # Move to archive if the date is not today
        if today != data_arquivo:
            source_path = os.path.join(base_directory, file_name)
            destination_path = os.path.join(archive_directory, file_name)
            os.rename(source_path, destination_path)
            print(f"Archived: {file_name}")


def download_file(url, file_name):
    response = requests.get(url, stream=True)
    with open(file_name, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)


def generate_csv_base(df, path_file_base):
    # organizar o arquivo base por dt_referencia
    df = pd.read_csv(path_file_base, sep=";")
    df = df.sort_values("dt_referencia")
    # set the index
    df.set_index("dt_referencia", inplace=True)
    df.to_csv(path_file_base, sep=";")


def generate_xlsx_base(df, path_saida):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(path_saida, engine="xlsxwriter")
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name="Sheet1")
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def get_dados(base_url):
    dados = [
        {
            "base_name": "acucar_cristal_sao_paulo",
            "url": base_url + "acucar.aspx?id=53",
        },
        {"base_name": "algodao_8_dias", "url": base_url + "algodao.aspx?id=54"},
        {"base_name": "arroz_em_casca", "url": base_url + "arroz.aspx?id=91"},
        {"base_name": "bezerro_ms", "url": base_url + "bezerro.aspx?id=8"},
        {"base_name": "boi_gordo", "url": base_url + "boi-gordo.aspx?id=2"},
        {"base_name": "cafe_arabica", "url": base_url + "cafe.aspx?id=23"},
        {"base_name": "cafe_robusta", "url": base_url + "cafe.aspx?id=24"},
        {
            "base_name": "etanol_hidratado_outros_fins",
            "url": base_url + "etanol.aspx?id=85",
        },
        {"base_name": "etanol_hidratado", "url": base_url + "etanol.aspx?id=103"},
        {"base_name": "etanol_anidro", "url": base_url + "etanol.aspx?id=104"},
        {"base_name": "frango_congelado", "url": base_url + "frango.aspx?id=181"},
        {"base_name": "frango_resfriado", "url": base_url + "frango.aspx?id=130"},
        {"base_name": "leite_liquido", "url": base_url + "leite.aspx?id=leitel"},
        {"base_name": "leite_bruto", "url": base_url + "leite.aspx?id=leite"},
        {"base_name": "raiz_mandioca", "url": base_url + "mandioca.aspx?id=72"},
        {"base_name": "fecula_mandioca", "url": base_url + "mandioca.aspx?id=71"},
        {"base_name": "milho", "url": base_url + "milho.aspx?id=77"},
        {"base_name": "ovos_produto_posto", "url": base_url + "ovos.aspx?id=158"},
        {"base_name": "ovos_produto_a_retirar", "url": base_url + "ovos.aspx?id=159"},
        {"base_name": "soja_parana", "url": base_url + "soja.aspx?id=12"},
        {"base_name": "soja_paranagua", "url": base_url + "soja.aspx?id=92"},
        {"base_name": "suino_vivo", "url": base_url + "suino.aspx?id=129"},
        {"base_name": "trigo_parana", "url": base_url + "trigo.aspx?id=178"},
    ]
    return dados


def main():
    # Remove old files
    archive_old_files()

    # Directory and file path setup
    base_directory = r"C:\Users\Max\Desktop\Max_Data_science\ANP_Gasolina"

    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    path_file_base = os.path.join(base_directory, f"preco_esalq_{today}.csv")
    base_url = "https://www.cepea.esalq.usp.br/br/indicador/series/"
    dados = get_dados(base_url)

    # Download and process files
    for dado in dados:
        name_file = dado["base_name"] + "_" + today + ".xls"
        path_file = os.path.join("downloads", name_file)
        print("Processing:", path_file)

        # Download the file if it does not exist
        if not os.path.exists(path_file):
            download_file(dado["url"], path_file)

        # Correct any read issues
        data = pyexcel_xls.get_data(path_file)
        pyexcel_xls.save_data(path_file, data)

        ignore_import_list = [
            "suino_vivo",
            "leite_liquido",
            "leite_bruto",
            "ovos_produto_posto",
            "ovos_produto_a_retirar",
            "raiz_mandioca",
            "fecula_mandioca",
        ]
        if dado["base_name"] in ignore_import_list:
            continue

        df = pd.read_excel(path_file, sheet_name="Plan 1", skiprows=3)
        df["no_produto"] = dado["base_name"]

        new_columns = {
            "À vista R$": "vr_real",
            "À vista US$": "vr_dolar",
            "Data": "dt_referencia",
        }
        df = df.rename(columns=new_columns)
        df["dt_referencia"] = pd.to_datetime(
            df["dt_referencia"], format="%d/%m/%Y", errors="ignore"
        )

        # Save to CSV base file with updated structure
        df[["dt_referencia", "no_produto", "vr_real", "vr_dolar"]].to_csv(
            path_file_base,
            mode="a",
            index=False,
            header=not os.path.exists(path_file_base),
            sep=";",
        )

    print(f"Crawl complete. Data saved to {path_file_base}")


if __name__ == "__main__":
    main()
